#!/usr/bin/env python3
"""
Module 3: Data Aggregation & Annotation
========================================
Fetches, cleans, and merges multiple psychotherapy-related datasets into a
unified Pandas DataFrame. Integrates:

  LOCAL (from TransferNow model zips):
    - GoEmotions   (54K utterances, 7 emotion classes)
    - IEMOCAP      (6.5K utterances, 151 sessions, speaker-labeled)
    - MELD         (12.3K utterances, 1,362 episodes, character-labeled)

  HUGGINGFACE HUB:
    - CounselChat  (nbertagnolli/counsel-chat)  — Q&A therapist sessions
    - EmpatheticDialogues (lighteval/empathetic_dialogues) — emotional conversations
    - Mental Health FAQ (tolu07/Mental_Health_FAQ) — Q&A format

  LOCAL MOCK CLINICAL DATA:
    - Text files in ./local_data/  (English + 5 South Indian languages)
    - Transcribed audio from Module 2 (transcriber.py output)

Unified schema:
    conversation_id : str    — groups utterances into sessions
    turn_id         : int    — within-session turn number
    speaker         : str    — "patient" or "therapist"
    text            : str    — PII-scrubbed utterance text
    emotion         : str    — emotion label (from source or model)
    source          : str    — provenance tag
    metadata        : dict   — extra fields (topic, confidence, language, etc.)

Usage:
  python dataset_builder.py --output curated_dataset.csv
  python dataset_builder.py --skip-hf --local-only  # only use local zip data
  python dataset_builder.py --mock-clinical-dir ./local_data/
  python dataset_builder.py --transcriptions ./transcriptions/  # from Module 2
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

# Module 1: PII Scrubbing
from pii_scrubber import PIIScrubber

# ──────────────────────────── Logging ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetBuilder")

# ──────────────────────────── Constants ────────────────────────────
MODELS_ZIP_DIR = Path(__file__).parent / "models"
PSYCHOTHERAPY_DIR = Path(__file__).parent

# HF Hub dataset identifiers
HF_DATASETS = {
    "counselchat":  {"name": "nbertagnolli/counsel-chat",        "split": "train"},
    "empathetic":   {"name": "lighteval/empathetic_dialogues",     "split": "train"},
    "medfaq":       {"name": "tolu07/Mental_Health_FAQ",           "split": "train"},
}

# Supported languages for mock clinical data
SUPPORTED_LANGS = {"en", "hi", "kn", "ta", "ml", "te"}

# Emotion normalization: map dataset-specific labels to unified set
UNIFIED_EMOTIONS = {
    # GoEmotions original labels -> unified
    "neutral": "neutral", "joy": "joy", "anger": "anger",
    "sadness": "sadness", "fear": "fear", "surprise": "surprise",
    "disgust": "disgust",
    # IEMOCAP labels
    "neu": "neutral", "ang": "anger", "hap": "joy",
    "sad": "sadness", "fea": "fear", "sur": "surprise", "fru": "frustration",
    "exc": "joy", "dis": "disgust",
    # MELD uses same as above
    # CounselChat / EmpatheticDialogues don't have labels — assign "unknown"
}

# Conversation format: Llama-3 / OpenAI ChatML style
SYSTEM_PROMPT = (
    "You are a helpful, empathetic psychotherapy assistant. "
    "The patient is sharing their thoughts and feelings. "
    "Respond with therapeutic understanding, validation, and evidence-based guidance."
)


# ─── Local Zip Data Loading ──────────────────────────────────────────
def load_from_zip(zip_name: str, csv_path: str) -> pd.DataFrame:
    """Load a CSV from within a model zip file."""
    zip_path = MODELS_ZIP_DIR / zip_name
    if not zip_path.exists():
        logger.warning(f"Zip not found: {zip_path}")
        return pd.DataFrame()

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_path) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
    logger.info(f"Loaded {csv_path} from {zip_name}: {len(df)} rows")
    return df


def load_goemotions() -> pd.DataFrame:
    """Load GoEmotions data from the emotion_model.zip."""
    df = load_from_zip("emotion_model.zip", "emotion_model/data/processed/goemotions_normalized.csv")
    if df.empty:
        return df
    df["source"] = "goemotions"
    df["emotion"] = df["emotion"].fillna("neutral").str.lower()
    logger.info(f"GoEmotions: {len(df)} utterances, emotions={df['emotion'].value_counts().to_dict()}")
    return df


def load_iemocap() -> pd.DataFrame:
    """Load IEMOCAP data from the emotion_model.zip."""
    df = load_from_zip("emotion_model.zip", "emotion_model/data/processed/iemocap_normalized.csv")
    if df.empty:
        return df
    df["source"] = "iemocap"
    df["emotion"] = df["emotion"].fillna("neutral").str.lower()
    # Map speaker labels
    df["speaker"] = df["speaker"].map({"Female": "patient", "Male": "therapist"})
    logger.info(f"IEMOCAP: {len(df)} utterances, {df['conversation_id'].nunique()} sessions")
    return df


def load_meld() -> pd.DataFrame:
    """Load MELD data from the emotion_model.zip."""
    df = load_from_zip("emotion_model.zip", "emotion_model/data/processed/meld_normalized.csv")
    if df.empty:
        return df
    df["source"] = "meld"
    df["emotion"] = df["emotion"].fillna("neutral").str.lower()
    # MELD speakers are character names; normalize to patient/therapist by alternating
    df["speaker"] = df.groupby("conversation_id")["turn_id"].transform(
        lambda x: x.apply(lambda i: "patient" if i % 2 == 0 else "therapist")
    )
    logger.info(f"MELD: {len(df)} utterances, {df['conversation_id'].nunique()} episodes")
    return df


# ─── HuggingFace Hub Data Loading ────────────────────────────────────
def load_counselchat(max_rows: int = None) -> pd.DataFrame:
    """
    Fetch CounselChat dataset from HF Hub.
    Each row is a Q&A pair: questionText (patient) -> answerText (therapist).
    """
    name = HF_DATASETS["counselchat"]["name"]
    logger.info(f"Fetching {name} from HF Hub...")
    try:
        import datasets
        ds = datasets.load_dataset(name, split="train", streaming=max_rows is not None)

        records = []
        count = 0
        for example in ds:
            q = (example.get("questionText") or example.get("questionTitle") or "").strip()
            a = (example.get("answerText") or "").strip()
            topic = example.get("topic") or example.get("topics") or "general"
            qid = example.get("questionID") or f"counselchat_{count}"

            if q and a and len(q) > 10 and len(a) > 10:
                # Strip HTML tags from answer
                a = re.sub(r"<[^>]+>", "", a).strip()
                records.append({
                    "conversation_id": str(qid),
                    "turn_id": 0,
                    "speaker": "patient",
                    "text": q,
                    "emotion": "unknown",
                    "source": "counselchat",
                    "metadata": {"topic": str(topic)},
                })
                records.append({
                    "conversation_id": str(qid),
                    "turn_id": 1,
                    "speaker": "therapist",
                    "text": a,
                    "emotion": "unknown",
                    "source": "counselchat",
                    "metadata": {"topic": str(topic)},
                })
                count += 1
                if max_rows and count >= max_rows:
                    break

        df = pd.DataFrame(records)
        logger.info(f"CounselChat: {len(df)} rows ({count} conversations)")
        return df

    except Exception as e:
        logger.error(f"Failed to load CounselChat: {e}")
        return pd.DataFrame()


def load_empathetic_dialogues(max_rows: int = None) -> pd.DataFrame:
    """
    Fetch EmpatheticDialogues from HF Hub (lighteval mirror, parquet-based).
    Format: conversational with conv_id, utterance_idx, speaker_idx, utterance.
    """
    name = HF_DATASETS["empathetic"]["name"]
    logger.info(f"Fetching {name} from HF Hub...")
    try:
        import datasets
        ds = datasets.load_dataset(name, split="train", streaming=max_rows is not None)

        records = []
        current_conv = None
        current_turns = []

        for example in ds:
            conv_id = str(example.get("conv_id", ""))
            utterance = (example.get("utterance") or "").strip()
            speaker_idx = example.get("speaker_idx", 0)
            context = example.get("context", "")
            prompt = example.get("prompt", "")

            if not utterance:
                continue

            # Group by conversation
            if conv_id != current_conv and current_turns:
                # Save completed conversation
                for i, (spk, utt) in enumerate(current_turns):
                    records.append({
                        "conversation_id": current_conv,
                        "turn_id": i,
                        "speaker": "patient" if spk == 0 else "therapist",
                        "text": utt,
                        "emotion": "unknown",
                        "source": "empathetic_dialogues",
                        "metadata": {
                            "context": str(context) if context else "",
                            "topic": str(prompt) if prompt else "",
                        },
                    })
                current_turns = []

            current_conv = conv_id
            current_turns.append((speaker_idx, utterance))

            if max_rows and len(records) >= max_rows * 2:
                break

        # Save last conversation
        if current_turns:
            for i, (spk, utt) in enumerate(current_turns):
                records.append({
                    "conversation_id": current_conv,
                    "turn_id": i,
                    "speaker": "patient" if spk == 0 else "therapist",
                    "text": utt,
                    "emotion": "unknown",
                    "source": "empathetic_dialogues",
                    "metadata": {"topic": str(prompt) if prompt else ""},
                })

        df = pd.DataFrame(records)
        logger.info(f"EmpatheticDialogues: {len(df)} rows ({df['conversation_id'].nunique()} conversations)")
        return df

    except Exception as e:
        logger.error(f"Failed to load EmpatheticDialogues: {e}")
        return pd.DataFrame()


def load_mental_health_faq(max_rows: int = None) -> pd.DataFrame:
    """
    Fetch Mental Health FAQ from HF Hub.
    Format: Questions and Answers pairs.
    """
    name = HF_DATASETS["medfaq"]["name"]
    logger.info(f"Fetching {name} from HF Hub...")
    try:
        import datasets
        ds = datasets.load_dataset(name, split="train", streaming=max_rows is not None)

        records = []
        count = 0
        for example in ds:
            q = str(example.get("question", example.get("Question", ""))).strip()
            a = str(example.get("answer", example.get("Answer", ""))).strip()

            if q and a and len(q) > 10 and len(a) > 10:
                records.append({
                    "conversation_id": f"faq_{count}",
                    "turn_id": 0,
                    "speaker": "patient",
                    "text": q,
                    "emotion": "unknown",
                    "source": "mental_health_faq",
                    "metadata": {},
                })
                records.append({
                    "conversation_id": f"faq_{count}",
                    "turn_id": 1,
                    "speaker": "therapist",
                    "text": a,
                    "emotion": "unknown",
                    "source": "mental_health_faq",
                    "metadata": {},
                })
                count += 1
                if max_rows and count >= max_rows:
                    break

        df = pd.DataFrame(records)
        logger.info(f"Mental Health FAQ: {len(df)} rows ({count} Q&A pairs)")
        return df

    except Exception as e:
        logger.warning(f"Failed to load Mental Health FAQ ({e}). Skipping.")
        return pd.DataFrame()


# ─── Mock Clinical Data Loading ──────────────────────────────────────
def load_mock_clinical_data(data_dir: str) -> pd.DataFrame:
    """
    Load local mock clinical data from text files.
    Expected structure:
      data_dir/
        session_001/
          transcript.txt    — "PATIENT: text\nTHERAPIST: text"
          language.txt      — ISO language code (en, hi, kn, etc.)
          metadata.json     — {topic, session_date, ...}

    Also reads transcribed audio results from Module 2.
    """
    if not data_dir or not Path(data_dir).exists():
        logger.info(f"Mock clinical data dir not found: {data_dir}")
        return pd.DataFrame()

    data_path = Path(data_dir)
    records = []

    # Find all session directories or text files
    session_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    txt_files = sorted(data_path.glob("*.txt"))

    for session_dir in session_dirs:
        records.extend(_parse_session_dir(session_dir))

    for txt_file in txt_files:
        records.extend(_parse_transcript_file(txt_file))

    if records:
        df = pd.DataFrame(records)
        logger.info(f"Mock clinical data: {len(df)} rows from {df['conversation_id'].nunique()} sessions")
        return df

    logger.warning("No mock clinical data found.")
    return pd.DataFrame()


def _parse_session_dir(session_dir: Path) -> List[Dict]:
    """Parse a session directory with transcript.txt and metadata."""
    records = []
    transcript_path = session_dir / "transcript.txt"
    meta_path = session_dir / "metadata.json"
    lang_path = session_dir / "language.txt"

    if not transcript_path.exists():
        return records

    session_id = session_dir.name

    # Load metadata
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    language = "en"
    if lang_path.exists():
        language = lang_path.read_text().strip()

    # Parse transcript: "PATIENT: text\nTHERAPIST: text\n..."
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    turn_id = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Parse speaker prefix
        match = re.match(r"^(PATIENT|THERAPIST|PATIENT_|THERAPIST_)[\s:]+(.+)", line, re.IGNORECASE)
        if match:
            speaker_raw = match.group(1).upper().replace("_", "")
            text = match.group(2).strip()
            speaker = "patient" if "PATIENT" in speaker_raw else "therapist"
        else:
            # No speaker prefix — alternate
            speaker = "patient" if turn_id % 2 == 0 else "therapist"
            text = line

        records.append({
            "conversation_id": session_id,
            "turn_id": turn_id,
            "speaker": speaker,
            "text": text,
            "emotion": "unknown",
            "source": "mock_clinical",
            "metadata": {"language": language, **metadata},
        })
        turn_id += 1

    return records


def _parse_transcript_file(txt_file: Path) -> List[Dict]:
    """Parse a standalone transcript file (from transcriber output or manual)."""
    records = []
    transcript_path = txt_file

    # If it's a JSON (from transcriber.py), parse segments
    if txt_file.suffix == ".json":
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            session_id = txt_file.stem
            for i, seg in enumerate(data.get("segments", [])):
                speaker = seg.get("speaker", "patient" if i % 2 == 0 else "therapist")
                records.append({
                    "conversation_id": session_id,
                    "turn_id": i,
                    "speaker": "patient" if speaker == "patient" else "therapist",
                    "text": seg.get("text", "").strip(),
                    "emotion": "unknown",
                    "source": "transcribed_audio",
                    "metadata": {"language": data.get("detected_language", "en")},
                })
        except Exception as e:
            logger.warning(f"Could not parse {txt_file}: {e}")
    elif txt_file.suffix == ".txt":
        records = _parse_session_dir(txt_file.parent / txt_file.stem)  # fallback

    return records


# ─── Transcript Integration ──────────────────────────────────────────
def load_transcriptions(transcription_dir: str) -> pd.DataFrame:
    """Load transcriptions produced by transcriber.py."""
    if not transcription_dir or not Path(transcription_dir).exists():
        return pd.DataFrame()

    trans_path = Path(transcription_dir)
    records = []

    # Look for .json files (transcriber output format)
    for json_file in sorted(trans_path.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            session_id = json_file.stem
            for i, seg in enumerate(data.get("segments", [])):
                speaker = seg.get("speaker", "patient" if i % 2 == 0 else "therapist")
                records.append({
                    "conversation_id": session_id,
                    "turn_id": i,
                    "speaker": "patient" if speaker == "patient" else "therapist",
                    "text": seg.get("text", "").strip(),
                    "emotion": "unknown",
                    "source": "transcribed_audio",
                    "metadata": {"language": data.get("detected_language", "en")},
                })
        except Exception as e:
            logger.warning(f"Could not parse transcription {json_file}: {e}")

    if records:
        df = pd.DataFrame(records)
        logger.info(f"Transcriptions loaded: {len(df)} utterances across {df['conversation_id'].nunique()} sessions")
        return df

    return pd.DataFrame()


# ─── Merge & Format ──────────────────────────────────────────────────
def merge_all_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge all DataFrames into a unified schema."""
    valid_dfs = [df for df in dfs if not df.empty]
    if not valid_dfs:
        logger.error("No data to merge!")
        return pd.DataFrame()

    # Ensure all DataFrames have the unified schema columns
    unified_cols = ["conversation_id", "turn_id", "speaker", "text",
                    "emotion", "source", "metadata"]

    merged = pd.concat(valid_dfs, ignore_index=True)
    if "metadata" not in merged.columns:
        merged["metadata"] = [{} for _ in range(len(merged))]
    else:
        merged["metadata"] = merged["metadata"].apply(lambda x: x if isinstance(x, dict) else {})
    merged = merged[[c for c in unified_cols if c in merged.columns]]

    # Ensure required columns
    for col in ["conversation_id", "turn_id", "speaker", "text", "emotion", "source"]:
        if col not in merged.columns:
            merged[col] = "unknown" if col == "emotion" else 0 if col == "turn_id" else ""

    # Deduplicate by text
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    if len(merged) < before:
        logger.info(f"Removed {before - len(merged)} duplicate utterances")

    logger.info(f"Merged dataset: {len(merged)} utterances from {merged['source'].value_counts().to_dict()}")

    return merged


def filter_and_sample(df: pd.DataFrame, target_conversations: int = 1500) -> pd.DataFrame:
    """
    Filter to highest-quality therapeutic conversations.
    Quality criteria:
      1. Must have therapist response (patient → therapist turn structure)
      2. Minimum utterance length (filter spam/short messages)
      3. Prefer conversations with emotion labels
      4. Balance across emotion categories
    """
    logger.info(f"Filtering {len(df)} utterances to ~{target_conversations} conversations...")

    # Group into conversations and assess quality
    conv_groups = df.groupby("conversation_id")
    quality_scores = []

    for conv_id, group in conv_groups:
        group = group.sort_values("turn_id")
        score = 0
        utts = group["text"].tolist()

        # Criterion 1: must have at least 2 turns (patient + therapist)
        if len(group) < 2:
            continue

        # Criterion 2: check for patient/therapist structure
        speakers = group["speaker"].tolist()
        has_patient = "patient" in speakers
        has_therapist = "therapist" in speakers
        if has_patient and has_therapist:
            score += 2

        # Criterion 3: utterance quality (length)
        min_len = group["text"].str.len().min()
        avg_len = group["text"].str.len().mean()
        if min_len > 10:
            score += 1
        if avg_len > 50:
            score += 1

        # Criterion 4: emotion labels present
        has_emotions = group["emotion"].notna().sum()
        if has_emotions > 0:
            score += 1

        # Criterion 5: not from low-quality sources (filter spammy patterns)
        if group["source"].iloc[0] == "goemotions":
            # GoEmotions is Reddit data — filter out very short utterances
            if avg_len > 30:
                score += 1
        elif group["source"].iloc[0] in ("counselchat", "empathetic_dialogues", "mock_clinical", "transcribed_audio"):
            score += 1

        quality_scores.append((conv_id, score, len(group)))

    # Sort by quality score and select top conversations
    quality_scores.sort(key=lambda x: (-x[1], -x[2]))
    selected_conv_ids = [q[0] for q in quality_scores[:target_conversations]]

    filtered = df[df["conversation_id"].isin(selected_conv_ids)].copy()
    logger.info(f"Filtered to {len(selected_conv_ids)} conversations ({len(filtered)} utterances)")

    return filtered


def format_for_finetuning(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert filtered DataFrame to Llama-3 / ChatML conversation format
    for LoRA fine-tuning.

    Schema (JSONL, one per line):
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "...", "metadata": {"emotion": "...", "speaker": "..."}},
          {"role": "assistant", "content": "...", "metadata": {"emotion": "..."}},
          ...
        ],
        "metadata": {
          "source": "...",
          "conversation_id": "...",
          "num_turns": N
        }
      }
    """
    records = []

    for conv_id, group in df.groupby("conversation_id"):
        group = group.sort_values("turn_id")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for _, row in group.iterrows():
            # Map speaker to role
            if row["speaker"] == "patient":
                role = "user"
            elif row["speaker"] == "therapist":
                role = "assistant"
            else:
                role = "user"  # default

            content = row["text"]
            msg = {"role": role, "content": content}

            # Inject emotion/intent metadata
            if row["emotion"] and row["emotion"] != "unknown":
                msg["metadata"] = {
                    "emotion": row["emotion"],
                    "source": row["source"],
                }

            messages.append(msg)

        # Only include conversations with at least one user + one assistant
        has_user = any(m["role"] == "user" for m in messages[1:])  # skip system
        has_assistant = any(m["role"] == "assistant" for m in messages[1:])
        if not (has_user and has_assistant):
            continue

        records.append({
            "messages": messages,
            "metadata": {
                "source": str(group["source"].iloc[0]),
                "conversation_id": str(conv_id),
                "num_turns": len(messages) - 1,  # exclude system
            },
        })

    logger.info(f"Formatted {len(records)} conversations for fine-tuning")
    return records


# ─── Main Pipeline ───────────────────────────────────────────────────
def build_dataset(
    output_path: str = None,
    skip_hf: bool = False,
    local_only: bool = False,
    mock_clinical_dir: str = None,
    transcription_dir: str = None,
    target_conversations: int = 1500,
    max_hf_rows: int = 5000,  # limit HF downloads for memory efficiency
) -> pd.DataFrame:
    """
    Main pipeline: load all data sources, merge, filter, and format.
    """
    t0 = time.time()
    all_dfs = []

    # 1. Local zip data (GoEmotions, IEMOCAP, MELD)
    logger.info("=" * 50)
    logger.info("Step 1: Loading local data from model zips")
    logger.info("=" * 50)

    all_dfs.append(load_goemotions())
    all_dfs.append(load_iemocap())
    all_dfs.append(load_meld())

    # 2. HuggingFace Hub data
    if not skip_hf and not local_only:
        logger.info("=" * 50)
        logger.info("Step 2: Fetching HuggingFace Hub datasets")
        logger.info("=" * 50)

        all_dfs.append(load_counselchat(max_rows=max_hf_rows))
        all_dfs.append(load_empathetic_dialogues(max_rows=max_hf_rows))
        all_dfs.append(load_mental_health_faq(max_rows=max_hf_rows))

    # 3. Mock clinical data
    if mock_clinical_dir and not local_only:
        logger.info("=" * 50)
        logger.info("Step 3: Loading local mock clinical data")
        logger.info("=" * 50)
        all_dfs.append(load_mock_clinical_data(mock_clinical_dir))

    # 4. Transcriptions from Module 2
    if transcription_dir and not local_only:
        logger.info("=" * 50)
        logger.info("Step 4: Loading transcribed audio data")
        logger.info("=" * 50)
        all_dfs.append(load_transcriptions(transcription_dir))

    # 5. Merge
    logger.info("=" * 50)
    logger.info("Step 5: Merging all data sources")
    logger.info("=" * 50)
    merged = merge_all_dataframes(all_dfs)

    if merged.empty:
        logger.error("No data loaded! Check data sources.")
        return merged

    # 6. PII Scrubbing (Module 1 integration)
    logger.info("=" * 50)
    logger.info("Step 6: Scrubbing PII from merged text")
    logger.info("=" * 50)
    scrubber = PIIScrubber()
    before_nulls = merged["text"].isna().sum()
    merged["text"] = merged["text"].apply(
        lambda x: scrubber.scrub(x) if pd.notna(x) and isinstance(x, str) else x
    )
    after_nulls = merged["text"].isna().sum()
    logger.info(f"PII scrubbing complete (text nulls: {before_nulls} -> {after_nulls})")

    # 7. Filter and sample
    logger.info("=" * 50)
    logger.info("Step 7: Filtering and sampling to high-quality conversations")
    logger.info("=" * 50)
    filtered = filter_and_sample(merged, target_conversations)

    # Step 8: Format for LLM fine-tuning
    logger.info("=" * 50)
    logger.info("Step 8: Formatting for LLM fine-tuning (ChatML/Llama-3)")
    logger.info("=" * 50)
    formatted = format_for_finetuning(filtered)

    # Step 9: Save intermediate CSV
    if output_path:
        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Save merged DataFrame as CSV
        merged_csv = str(csv_path).replace(".csv", "_merged.csv")
        merged.to_csv(merged_csv, index=False, encoding="utf-8")
        logger.info(f"Merged dataset saved to: {merged_csv}")

        # Save filtered DataFrame as CSV
        filtered.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Filtered dataset saved to: {output_path}")

        # Save formatted JSONL
        jsonl_path = str(csv_path).replace(".csv", ".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in formatted:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Formatted JSONL saved to: {jsonl_path}")

    elapsed = time.time() - t0
    logger.info(f"\nPipeline complete in {elapsed:.1f}s")
    logger.info(f"  Total utterances merged: {len(merged)}")
    logger.info(f"  Filtered utterances: {len(filtered)}")
    logger.info(f"  Conversations (formatted): {len(formatted)}")
    logger.info(f"  Sources: {merged['source'].value_counts().to_dict()}")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and format psychotherapy datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_builder.py --output data/filtered_dataset.csv
  python dataset_builder.py --skip-hf --output data/local_only.csv
  python dataset_builder.py --mock-clinical-dir ./local_data/ --transcriptions ./transcriptions/
        """,
    )
    parser.add_argument("--output", "-o", type=str,
                        default="data/curated_dataset.csv",
                        help="Output CSV path for filtered dataset.")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace Hub downloads.")
    parser.add_argument("--local-only", action="store_true",
                        help="Only use local zip data (no HF Hub, no mock data).")
    parser.add_argument("--mock-clinical-dir", type=str, default="./local_data/",
                        help="Directory containing local mock clinical sessions.")
    parser.add_argument("--transcriptions", type=str, default=None,
                        help="Directory with transcriber.py JSON output.")
    parser.add_argument("--target-conversations", type=int, default=1500,
                        help="Target number of curated conversations (default: 1500).")
    parser.add_argument("--max-hf-rows", type=int, default=5000,
                        help="Max rows per HF dataset (memory efficiency).")

    args = parser.parse_args()
    build_dataset(
        output_path=args.output,
        skip_hf=args.skip_hf,
        local_only=args.local_only,
        mock_clinical_dir=args.mock_clinical_dir,
        transcription_dir=args.transcriptions,
        target_conversations=args.target_conversations,
        max_hf_rows=args.max_hf_rows,
    )


if __name__ == "__main__":
    main()
