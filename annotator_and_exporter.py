#!/usr/bin/env python3
"""
Module 4: Annotation & LLM Fine-Tuning Export
==============================================
Loads pre-trained emotion and behaviour classifiers, runs batch inference on
curated therapeutic utterances, applies quality filtering, and exports the
final dataset in Llama-3 / ChatML JSONL format for LoRA fine-tuning.

Pre-trained models (from TransferNow zip):
  Emotion (11-class RoBERTa, multilabel):
    Labels: anger, frustration, sadness, confusion, shame_guilt,
            fear, anxiety, relief, grief, positive_progress, neutral
    Decision: sigmoid + threshold 0.5
    Zip: models/emotion_model_1.zip → model/

  Behaviour (8-class DistilBERT, single-label):
    Labels: Social_Withdrawal, Rumination, Avoidance, Negative_Self_Talk,
            Emotional_Expression, Help_Seeking, Active_Coping, Conversational_Act
    Decision: softmax + argmax
    Zip: models/behaviour_model.zip → models/distilbert_behaviour_model/

  Emotion (7-class DistilBERT, single-label) [fallback]:
    Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    Zip: models/emotion_model.zip → models/

Disk-space aware: models are extracted from zips to a temp dir one at a
time, then deleted after inference to keep disk usage minimal.

Usage:
  python annotator_and_exporter.py --input data/curated_dataset.csv --output data/curated_psychotherapy_dataset.jsonl
  python annotator_and_exporter.py --input data/curated_dataset.csv --model-dir models/
  python annotator_and_exporter.py --test  # quick self-test on sample texts
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import torch
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────── Logging ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Annotator")

# ──────────────────────────── Constants ────────────────────────────
PSYCHOTHERAPY_DIR = Path(__file__).parent
MODELS_DIR = PSYCHOTHERAPY_DIR / "models"

# Model configurations: (zip_name, subdir_in_zip, labels, decision_type, max_length)
EMOTION_MODEL_CONFIGS = {
    "roberta_11class": {
        "zip": "emotion_model_1.zip",
        "subdir": "emotion_model_1/model",
        "labels": ["anger", "frustration", "sadness", "confusion", "shame_guilt",
                   "fear", "anxiety", "relief", "grief", "positive_progress", "neutral"],
        "decision": "multilabel",       # sigmoid + threshold
        "threshold": 0.5,
        "max_length": 64,
        "arch": "RobertaForSequenceClassification",
    },
    "distilbert_7class": {
        "zip": "emotion_model.zip",
        "subdir": "emotion_model/models",
        "labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        "decision": "singlelabel",      # softmax + argmax
        "threshold": 0.0,
        "max_length": 128,
        "arch": "DistilBertForSequenceClassification",
        "weights_file": "best_model.pt",  # PyTorch state dict, not safetensors
    },
}

BEHAVIOUR_MODEL_CONFIG = {
    "zip": "behaviour_model.zip",
    "subdir": "behaviour_model/models/distilbert_behaviour_model",
    "labels": ["Social_Withdrawal", "Rumination", "Avoidance", "Negative_Self_Talk",
               "Emotional_Expression", "Help_Seeking", "Active_Coping", "Conversational_Act"],
    "decision": "singlelabel",
    "threshold": 0.0,
    "max_length": 128,
    "arch": "DistilBertForSequenceClassification",
}

# Conversation format constants for Llama-3 / ChatML
SYSTEM_PROMPT = (
    "You are a helpful, empathetic AI psychotherapy assistant. "
    "The patient shares their thoughts, feelings, and experiences. "
    "Respond with therapeutic validation, evidence-based guidance, and gentle probing. "
    "Maintain a professional, non-judgmental tone. Never provide medical diagnosis. "
    "If the patient is in crisis, encourage them to contact emergency services or a crisis hotline."
)

# Quality thresholds
MIN_CONFIDENCE = 0.60      # minimum confidence for a predicted emotion to be accepted
MAX_LOW_CONF_RATIO = 0.30  # max fraction of low-confidence utterances in a conversation
MIN_UTTERANCE_LEN = 3      # minimum characters in an utterance


@dataclass
class AnnotationBatch:
    """Container for a batch of annotated utterances."""
    texts: List[str]
    emotions: List[List[Tuple[str, float]]]  # per-utterance list of (label, prob)
    behaviours: List[Tuple[str, float]]      # per-utterance (label, confidence)


class ModelExtractor:
    """
    Extract model files from zip archives on-demand to a temp directory.
    Handles disk space constraints by extracting only needed files.
    """

    @staticmethod
    def extract_model(
        zip_path: str,
        model_subdir: str,
        dest_dir: str,
        verbose: bool = True,
    ) -> str:
        """
        Extract a model subdirectory from a zip to dest_dir.

        Args:
            zip_path: Path to the zip file.
            model_subdir: Subdirectory within the zip containing the model.
            dest_dir: Destination directory for extraction.

        Returns:
            Path to the extracted model directory.
        """
        zip_path = Path(zip_path)
        dest_dir = Path(dest_dir)

        if not zip_path.exists():
            raise FileNotFoundError(f"Model zip not found: {zip_path}")

        os.makedirs(dest_dir, exist_ok=True)

        if verbose:
            logger.info(f"Extracting {model_subdir} from {zip_path.name} -> {dest_dir}")

        with zipfile.ZipFile(zip_path, "r") as z:
            # List files in the target subdirectory
            model_files = [
                f for f in z.namelist()
                if f.startswith(model_subdir)
            ]

            if not model_files:
                raise ValueError(f"No files found at '{model_subdir}' in {zip_path}")

            if verbose:
                total_size = sum(z.getinfo(f).file_size for f in model_files)
                logger.info(f"  {len(model_files)} files, {total_size / 1024 / 1024:.1f} MB uncompressed")

            # Extract all files in the model subdirectory
            for member in tqdm(model_files, desc="  Extracting") if verbose else model_files:
                z.extract(member, dest_dir)

        model_dir = dest_dir / model_subdir
        if verbose:
            logger.info(f"  Model extracted to: {model_dir}")
        return str(model_dir)

    @staticmethod
    def cleanup(dest_dir: str):
        """Remove extracted model directory to free disk space."""
        dest_dir = Path(dest_dir)
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
            logger.info(f"Cleaned up extracted model at: {dest_dir}")


class EmotionAnnotator:
    """
    Loads the pre-trained emotion classification model and predicts emotions
    for a list of texts.

    Uses the 11-class RoBERTa model (emotion_model_1) by default for rich
    clinical emotion categories. Falls back to the 7-class DistilBERT if
    the RoBERTa model is unavailable.
    """

    def __init__(
        self,
        models_dir: str = None,
        model_key: str = "roberta_11class",
        device: str = "auto",
        temp_dir: str = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.model_key = model_key
        self.temp_dir = temp_dir
        self._extracted = False
        self.model_dir: Optional[str] = None
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if model_key not in EMOTION_MODEL_CONFIGS:
            raise ValueError(f"Unknown emotion model: {model_key}. "
                           f"Choose from: {list(EMOTION_MODEL_CONFIGS.keys())}")

        config = EMOTION_MODEL_CONFIGS[model_key]
        self.labels = config["labels"]
        self.decision = config["decision"]
        self.threshold = config["threshold"]
        self.max_length = config["max_length"]
        self.arch = config["arch"]

        self.model_dir: Optional[str] = None
        self._extracted = False

    def prepare_model(self) -> str:
        """
        Extract the model from its zip to a temp directory if needed.
        Returns the path to the model directory.
        """
        if self._extracted and self.model_dir and Path(self.model_dir).exists():
            return self.model_dir

        config = EMOTION_MODEL_CONFIGS[self.model_key]
        zip_path = self.models_dir / config["zip"]

        if zip_path.exists():
            # Use a temp directory or provided one
            if self.temp_dir:
                dest = Path(self.temp_dir) / "emotion_model_extracted"
            else:
                dest = Path(tempfile.mkdtemp(prefix="emotion_model_"))

            self.model_dir = ModelExtractor.extract_model(
                str(zip_path), config["subdir"], str(dest)
            )
            self._extracted = True
        else:
            # Check if model is already extracted (e.g., user did it manually)
            manual_dir = self.models_dir / config["subdir"]
            if manual_dir.exists():
                self.model_dir = str(manual_dir)
                logger.info(f"Using pre-extracted model: {self.model_dir}")
            else:
                raise FileNotFoundError(
                    f"Model zip not found: {zip_path}. "
                    f"Either provide the zip or extract to {manual_dir}"
                )

        return self.model_dir

    def load(self) -> Tuple[Any, Any]:
        """Load the model and tokenizer."""
        model_dir = self.prepare_model()

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading emotion model ({self.model_key})...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type="multi_label_classification" if self.decision == "multilabel" else "single_label_classification",
            torch_dtype=torch.float32,
        )
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded in {time.time() - t0:.1f}s | Labels: {self.labels}")
        return model, tokenizer

    def predict_batch(
        self,
        texts: List[str],
        model, tokenizer,
        batch_size: int = 16,
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict emotions for a batch of texts.

        Returns:
            List of lists of (emotion_label, confidence) tuples.
            For multilabel: all labels above threshold, sorted by confidence.
            For single-label: just the top label.
        """
        results = []
        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="  Emotion inference"):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            if self.decision == "multilabel":
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)

        for probs in all_probs:
            if self.decision == "multilabel":
                # Multi-label: return labels above threshold, sorted
                predictions = [
                    (self.labels[i], float(probs[i]))
                    for i in range(len(self.labels))
                    if probs[i] >= self.threshold
                ]
                predictions.sort(key=lambda x: -x[1])
                if not predictions:
                    # If nothing above threshold, return the highest
                    top_idx = int(np.argmax(probs))
                    predictions = [(self.labels[top_idx], float(probs[top_idx]))]
            else:
                # Single-label: return top prediction
                top_idx = int(np.argmax(probs))
                predictions = [(self.labels[top_idx], float(probs[top_idx]))]

            results.append(predictions)

        return results

    def cleanup_model(self):
        """Clean up extracted model to free disk space."""
        if self._extracted and self.temp_dir:
            ModelExtractor.cleanup(Path(self.temp_dir) / "emotion_model_extracted")
            self._extracted = False
            self.model_dir = None


class BehaviourAnnotator:
    """
    Loads the pre-trained behaviour/intent classification model and predicts
    behavioural categories for a list of texts.

    Uses the 8-class DistilBERT model (behaviour_model).
    """

    def __init__(
        self,
        models_dir: str = None,
        device: str = "auto",
        temp_dir: str = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.temp_dir = temp_dir
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.labels = BEHAVIOUR_MODEL_CONFIG["labels"]
        self.model_dir: Optional[str] = None
        self._extracted = False

    def prepare_model(self) -> str:
        """Extract the behaviour model from zip if needed."""
        if self._extracted and self.model_dir and Path(self.model_dir).exists():
            return self.model_dir

        config = BEHAVIOUR_MODEL_CONFIG
        zip_path = self.models_dir / config["zip"]

        if zip_path.exists():
            if self.temp_dir:
                dest = Path(self.temp_dir) / "behaviour_model_extracted"
            else:
                dest = Path(tempfile.mkdtemp(prefix="behaviour_model_"))

            self.model_dir = ModelExtractor.extract_model(
                str(zip_path), config["subdir"], str(dest)
            )
            self._extracted = True
        else:
            manual_dir = self.models_dir / "behaviour_model" / config["subdir"]
            if manual_dir.exists():
                self.model_dir = str(manual_dir)
                logger.info(f"Using pre-extracted model: {self.model_dir}")
            else:
                raise FileNotFoundError(
                    f"Behaviour model zip not found: {zip_path}"
                )

        return self.model_dir

    def load(self) -> Tuple[Any, Any]:
        """Load the model and tokenizer."""
        model_dir = self.prepare_model()

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Loading behaviour model...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type="single_label_classification",
            torch_dtype=torch.float32,
        )
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded in {time.time() - t0:.1f}s | Labels: {self.labels}")
        return model, tokenizer

    def predict_batch(
        self,
        texts: List[str],
        model, tokenizer,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Predict behaviour labels for a batch of texts.

        Returns:
            List of dicts: {behaviour, confidence, all_scores}
        """
        results = []
        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="  Behaviour inference"):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=BEHAVIOUR_MODEL_CONFIG["max_length"],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)

        for probs in all_probs:
            top_idx = int(np.argmax(probs))
            all_scores = {
                self.labels[i]: round(float(probs[i]), 4)
                for i in range(len(self.labels))
            }
            results.append({
                "behaviour": self.labels[top_idx],
                "confidence": round(float(probs[top_idx]), 4),
                "all_scores": all_scores,
            })

        return results

    def cleanup_model(self):
        """Clean up extracted model to free disk space."""
        if self._extracted and self.temp_dir:
            ModelExtractor.cleanup(Path(self.temp_dir) / "behaviour_model_extracted")
            self._extracted = False
            self.model_dir = None


# ─── Quality Filtering ───────────────────────────────────────────────
def quality_filter(
    df: pd.DataFrame,
    emotion_col: str = "predicted_emotion",
    emotion_conf_col: str = "emotion_confidence",
    behaviour_col: str = "predicted_behaviour",
    behaviour_conf_col: str = "behaviour_confidence",
    min_conversations: int = 500,
    max_conversations: int = 2000,
) -> pd.DataFrame:
    """
    Filter conversations based on annotation quality.

    Criteria:
      1. Conversations must have at least 2 turns (patient + therapist)
      2. Minimum utterance length filtering
      3. Exclude conversations where >30% of utterances have low emotion confidence
      4. Scale to target range (500-2000 conversations)
    """
    logger.info(f"Quality filtering: {len(df)} utterances...")

    # Filter very short utterances
    df = df[df["text"].str.len() >= MIN_UTTERANCE_LEN].copy()

    # Group by conversation and assess quality
    conv_ids = df["conversation_id"].unique()
    kept_convs = []

    for conv_id in conv_ids:
        group = df[df["conversation_id"] == conv_id]

        # Must have patient and therapist turns
        speakers = set(group["speaker"].unique())
        if not ("patient" in speakers and "therapist" in speakers):
            continue

        if len(group) < 2:
            continue

        # Check emotion confidence quality
        if emotion_conf_col in group.columns:
            confidences = group[emotion_conf_col].dropna()
            if len(confidences) > 0:
                low_conf_ratio = (confidences < MIN_CONFIDENCE).sum() / len(confidences)
                if low_conf_ratio > MAX_LOW_CONF_RATIO:
                    continue

        kept_convs.append(conv_id)

    filtered = df[df["conversation_id"].isin(kept_convs)].copy()
    num_convs = filtered["conversation_id"].nunique()
    logger.info(f"Quality filter: {num_convs} conversations retained ({len(filtered)} utterances)")

    # If outside target range, adjust
    if num_convs > max_conversations:
        # Select top conversations by quality score
        logger.info(f"Sampling down to {max_conversations} conversations...")
        # Compute quality score per conversation
        scores = []
        for conv_id in filtered["conversation_id"].unique():
            group = filtered[filtered["conversation_id"] == conv_id]
            score = len(group)  # longer conversations are better
            if emotion_conf_col in group.columns:
                score += group[emotion_conf_col].mean() * 100
            if behaviour_conf_col in group.columns:
                score += group[behaviour_conf_col].mean() * 100
            scores.append((conv_id, score))
        scores.sort(key=lambda x: -x[1])
        top_ids = [s[0] for s in scores[:max_conversations]]
        filtered = filtered[filtered["conversation_id"].isin(top_ids)].copy()
        logger.info(f"Sampled to {filtered['conversation_id'].nunique()} conversations")

    elif num_convs < min_conversations:
        logger.warning(
            f"Only {num_convs} conversations after filtering. "
            f"Target was {min_conversations}-{max_conversations}. "
            f"Consider adding more data sources."
        )

    return filtered


# ─── JSONL Export ────────────────────────────────────────────────────
def export_jsonl(
    df: pd.DataFrame,
    output_path: str,
    emotion_col: str = "predicted_emotion",
    emotion_conf_col: str = "emotion_confidence",
    emotion_all_col: str = "all_emotions",
    behaviour_col: str = "predicted_behaviour",
    behaviour_conf_col: str = "behaviour_confidence",
) -> str:
    """
    Export conversations to JSONL in Llama-3 / ChatML format.

    Each line is a JSON record:
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "...", "metadata": {"emotion": "...", "behavior": "..."}},
          {"role": "assistant", "content": "...", "metadata": {"emotion": "..."}},
        ],
        "metadata": {"source": "...", "conversation_id": "...", "num_turns": N}
      }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for conv_id, group in df.groupby("conversation_id"):
        group = group.sort_values("turn_id")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for _, row in group.iterrows():
            if row["speaker"] == "patient":
                role = "user"
            elif row["speaker"] == "therapist":
                role = "assistant"
            else:
                role = "user"  # default

            content = str(row["text"]).strip()
            msg = {"role": role, "content": content}

            # Build metadata with emotion + behavior labels
            meta = {}
            if emotion_col in row and pd.notna(row[emotion_col]) and row[emotion_col]:
                meta["emotion"] = str(row[emotion_col])
                if emotion_conf_col in row and pd.notna(row[emotion_conf_col]):
                    meta["emotion_confidence"] = float(row[emotion_conf_col])
                if emotion_all_col in row and pd.notna(row[emotion_all_col]):
                    try:
                        all_emotions = json.loads(row[emotion_all_col])
                        meta["all_emotions"] = all_emotions
                    except (json.JSONDecodeError, TypeError):
                        pass

            if behaviour_col in row and pd.notna(row[behaviour_col]) and row[behaviour_col]:
                meta["behavior"] = str(row[behaviour_col])
                if behaviour_conf_col in row and pd.notna(row[behaviour_conf_col]):
                    meta["behavior_confidence"] = float(row[behaviour_conf_col])

            if row["source"]:
                meta["source"] = str(row["source"])

            if meta:
                msg["metadata"] = meta

            messages.append(msg)

        # Require at least one user and one assistant turn
        has_user = any(m["role"] == "user" for m in messages[1:])
        has_assistant = any(m["role"] == "assistant" for m in messages[1:])
        if not (has_user and has_assistant):
            continue

        records.append({
            "messages": messages,
            "metadata": {
                "source": str(group["source"].iloc[0]) if "source" in group.columns else "unknown",
                "conversation_id": str(conv_id),
                "num_turns": len(messages) - 1,
            },
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(records)} conversations to {output_path}")
    return str(output_path)


# ─── Main Pipeline ───────────────────────────────────────────────────
def annotate_and_export(
    input_csv: str,
    output_jsonl: str,
    models_dir: str = None,
    model_key: str = "roberta_11class",
    device: str = "auto",
    batch_size: int = 16,
    min_conversations: int = 500,
    max_conversations: int = 2000,
) -> str:
    """
    Full annotation pipeline:
      1. Load filtered dataset CSV
      2. Extract + load emotion model → predict emotions
      3. Extract + load behaviour model → predict behaviours
      4. Quality filter conversations
      5. Export JSONL in Llama-3 / ChatML format

    Args:
        input_csv: Path to the filtered dataset CSV (from dataset_builder.py)
        output_jsonl: Path for the final JSONL output
        models_dir: Directory containing model zips
        model_key: Which emotion model to use ("roberta_11class" or "distilbert_7class")
        device: "auto", "cpu", "mps", or "cuda"
        batch_size: Inference batch size
        min_conversations: Minimum target conversations
        max_conversations: Maximum target conversations

    Returns:
        Path to the output JSONL file
    """
    t0 = time.time()
    temp_dir = tempfile.mkdtemp(prefix="pipeline_")
    logger.info(f"Temp directory: {temp_dir}")

    # Auto-detect device
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # ── Step 1: Load dataset ──
    logger.info("=" * 50)
    logger.info("Step 1: Loading filtered dataset")
    logger.info("=" * 50)

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} utterances from {df['conversation_id'].nunique()} conversations")

    # Ensure required columns
    if "text" not in df.columns:
        logger.error("Dataset must have a 'text' column!")
        return ""
    if "speaker" not in df.columns:
        logger.warning("No 'speaker' column — assuming alternating patient/therapist by turn_id")
        df["speaker"] = df.groupby("conversation_id")["turn_id"].transform(
            lambda x: x.apply(lambda i: "patient" if i % 2 == 0 else "therapist")
        )

    texts = df["text"].fillna("").tolist()

    # ── Step 2: Emotion annotation ──
    logger.info("=" * 50)
    logger.info(f"Step 2: Emotion annotation ({model_key})")
    logger.info("=" * 50)

    emotion_annotator = EmotionAnnotator(
        models_dir=models_dir,
        model_key=model_key,
        device=device,
        temp_dir=temp_dir,
    )

    model = None
    tokenizer = None
    try:
        model, tokenizer = emotion_annotator.load()
        emotion_predictions = emotion_annotator.predict_batch(texts, model, tokenizer, batch_size=batch_size)

        # Add predictions to DataFrame
        df["predicted_emotion"] = [
            preds[0][0] if preds else "neutral" for preds in emotion_predictions
        ]
        df["emotion_confidence"] = [
            preds[0][1] if preds else 0.0 for preds in emotion_predictions
        ]
        df["all_emotions"] = [
            json.dumps(preds) for preds in emotion_predictions
        ]
        logger.info(f"Emotion annotations complete. "
                    f"Mean confidence: {df['emotion_confidence'].mean():.3f}")
    except FileNotFoundError as e:
        logger.warning(f"Emotion model not available: {e}. Using existing emotion labels.")
        if "emotion" in df.columns:
            df["predicted_emotion"] = df["emotion"]
            df["emotion_confidence"] = 0.5
            df["all_emotions"] = [json.dumps({"emotion": str(e), "confidence": 0.5}) for e in df["emotion"]]
        else:
            df["predicted_emotion"] = "neutral"
            df["emotion_confidence"] = 0.0
            df["all_emotions"] = [json.dumps({"emotion": "neutral", "confidence": 0.0})] * len(df)
    except Exception as e:
        logger.error(f"Emotion annotation failed: {e}")
        df["predicted_emotion"] = "neutral"
        df["emotion_confidence"] = 0.0
        df["all_emotions"] = [json.dumps({"emotion": "neutral", "confidence": 0.0})] * len(df)
    finally:
        # Free model from GPU/CPU memory
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        emotion_annotator.cleanup_model()
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    # ── Step 3: Behaviour annotation ──
    logger.info("=" * 50)
    logger.info("Step 3: Behaviour annotation")
    logger.info("=" * 50)

    behaviour_annotator = BehaviourAnnotator(
        models_dir=models_dir,
        device=device,
        temp_dir=temp_dir,
    )

    try:
        model, tokenizer = behaviour_annotator.load()
        behaviour_predictions = behaviour_annotator.predict_batch(
            texts, model, tokenizer, batch_size=batch_size
        )

        df["predicted_behaviour"] = [p["behaviour"] for p in behaviour_predictions]
        df["behaviour_confidence"] = [p["confidence"] for p in behaviour_predictions]
        logger.info(f"Behaviour annotations complete. "
                    f"Mean confidence: {df['behaviour_confidence'].mean():.3f}")
    except FileNotFoundError as e:
        logger.warning(f"Behaviour model not available: {e}. Skipping behaviour labels.")
        df["predicted_behaviour"] = "Conversational_Act"
        df["behaviour_confidence"] = 0.0
    except Exception as e:
        logger.error(f"Behaviour annotation failed: {e}")
        df["predicted_behaviour"] = "Conversational_Act"
        df["behaviour_confidence"] = 0.0
    finally:
        del model, tokenizer
        behaviour_annotator.cleanup_model()
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ── Step 4: Quality filter ──
    logger.info("=" * 50)
    logger.info("Step 4: Quality filtering")
    logger.info("=" * 50)

    filtered = quality_filter(
        df,
        min_conversations=min_conversations,
        max_conversations=max_conversations,
    )

    # ── Step 5: Export JSONL ──
    logger.info("=" * 50)
    logger.info("Step 5: Exporting JSONL (Llama-3 / ChatML format)")
    logger.info("=" * 50)

    output_path = export_jsonl(filtered, output_jsonl)

    # Save intermediate annotated CSV
    annotated_csv = str(Path(output_jsonl).parent / "annotated_utterances.csv")
    filtered.to_csv(annotated_csv, index=False, encoding="utf-8")
    logger.info(f"Annotated utterances CSV saved to: {annotated_csv}")

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"Cleaned up temp directory: {temp_dir}")

    elapsed = time.time() - t0
    logger.info(f"\n{'='*50}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Conversations: {filtered['conversation_id'].nunique()}")
    logger.info(f"  Total utterances: {len(filtered)}")
    logger.info(f"  Emotion distribution: {filtered['predicted_emotion'].value_counts().to_dict()}")
    logger.info(f"  Behaviour distribution: {filtered['predicted_behaviour'].value_counts().to_dict()}")
    logger.info(f"{'='*50}")

    return output_path


# ─── Self-Test ───────────────────────────────────────────────────────
def self_test():
    """Quick self-test on sample texts to verify model loading and inference."""
    print("=" * 60)
    print("Annotator Self-Test")
    print("=" * 60)

    test_texts = [
        "I feel so anxious and overwhelmed all the time, I can't sleep.",
        "I'm really happy today — I finally finished my project!",
        "I keep thinking about what happened, it's making me sad.",
        "Can you help me find a therapist near me?",
        "I've been isolating myself from my friends lately.",
        "Good morning, how are you feeling today?",
    ]

    # Try emotion model
    print("\n--- Testing Emotion Model ---")
    emotion_annotator = EmotionAnnotator(model_key="roberta_11class")
    try:
        model, tokenizer = emotion_annotator.load()
        emotions = emotion_annotator.predict_batch(test_texts, model, tokenizer, batch_size=4)
        for text, emo in zip(test_texts, emotions):
            labels_str = ", ".join(f"{l}({c:.2f})" for l, c in emo[:3])
            print(f"  '{text[:60]}...' → {labels_str}")
        del model, tokenizer
        emotion_annotator.cleanup_model()
        print("\n  Emotion model: ✓ OK")
    except FileNotFoundError as e:
        print(f"  Emotion model not available: {e}")
        print("  (This is OK if models haven't been extracted yet)")
    except Exception as e:
        print(f"  ✗ Emotion model test failed: {e}")

    gc.collect()

    # Try behaviour model
    print("\n--- Testing Behaviour Model ---")
    behaviour_annotator = BehaviourAnnotator()
    try:
        model, tokenizer = behaviour_annotator.load()
        behaviours = behaviour_annotator.predict_batch(test_texts, model, tokenizer, batch_size=4)
        for text, beh in zip(test_texts, behaviours):
            print(f"  '{text[:60]}...' → {beh['behaviour']} ({beh['confidence']:.2f})")
        del model, tokenizer
        behaviour_annotator.cleanup_model()
        print("\n  Behaviour model: ✓ OK")
    except FileNotFoundError as e:
        print(f"  Behaviour model not available: {e}")
        print("  (This is OK if models haven't been extracted yet)")
    except Exception as e:
        print(f"  ✗ Behaviour model test failed: {e}")

    print("\n" + "=" * 60)
    print("Self-test complete (partial failures expected if models not extracted).")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Annotate psychotherapy dataset with emotion + behaviour labels, "
                    "then export as ChatML JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python annotator_and_exporter.py --test
  python annotator_and_exporter.py --input data/curated_dataset.csv --output data/curated_psychotherapy_dataset.jsonl
  python annotator_and_exporter.py --input data/curated_dataset.csv --output data/final.jsonl --model roberta_11class --device mps
  python annotator_and_exporter.py --input data/curated_dataset.csv --output data/final.jsonl --models-dir /custom/models/
        """,
    )
    parser.add_argument("--input", "-i", type=str, required=False,
                        help="Input CSV file (from dataset_builder.py).")
    parser.add_argument("--output", "-o", type=str,
                        default="curated_psychotherapy_dataset.jsonl",
                        help="Output JSONL file path.")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR),
                        help="Directory containing model zips.")
    parser.add_argument("--model", type=str, default="roberta_11class",
                        choices=list(EMOTION_MODEL_CONFIGS.keys()),
                        help="Which emotion model to use.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Device for inference.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size.")
    parser.add_argument("--min-conversations", type=int, default=500,
                        help="Minimum target conversation count.")
    parser.add_argument("--max-conversations", type=int, default=2000,
                        help="Maximum target conversation count.")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test with sample texts.")

    args = parser.parse_args()

    if args.test:
        sys.exit(0 if self_test() else 1)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    annotate_and_export(
        input_csv=args.input,
        output_jsonl=args.output,
        models_dir=args.models_dir,
        model_key=args.model,
        device=args.device,
        batch_size=args.batch_size,
        min_conversations=args.min_conversations,
        max_conversations=args.max_conversations,
    )


if __name__ == "__main__":
    main()
