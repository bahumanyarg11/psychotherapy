#!/usr/bin/env python3
"""
End-to-End Pipeline Runner
============================

Orchestrates the full psychotherapy dataset curation pipeline:

    1. (Optional) Transcribe local audio files  →  JSON transcripts
    2. Build & clean the unified dataset        →  intermediate.parquet
    3. Annotate emotions + behaviours           →  annotated DataFrame
    4. Export to JSONL for LLM fine-tuning      →  curated_psychotherapy_dataset.jsonl

Usage
─────
    python run_pipeline.py

    # With custom parameters
    python run_pipeline.py --target-size 2000 --skip-transcription --format chatml

    # Dry-run (no model loading, no file output)
    python run_pipeline.py --dry-run

    # Inspect available models
    python run_pipeline.py --list-models
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger("pipeline")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_models(transfer_zip: str, extract_to: str) -> dict[str, str]:
    """Extract the three inner model zips from the transfer archive.

    Returns a dict mapping model-role → extracted path.
    """
    transfer_p = Path(transfer_zip)
    if not transfer_p.exists():
        logger.warning("Transfer zip not found: %s — skipping model extraction", transfer_zip)
        return {}

    extract_p = Path(extract_to)
    extract_p.mkdir(parents=True, exist_ok=True)

    paths = {}
    with zipfile.ZipFile(transfer_p) as outer:
        inner_zips = {
            "emotion_model_1": "emotion_model_1.zip",
            "emotion_model": "emotion_model.zip",
            "behaviour_model": "behaviour_model.zip",
        }
        for role, inner_name in inner_zips.items():
            try:
                outer.extract(inner_name, extract_p)
                inner_path = extract_p / inner_name.replace(".zip", "")
                # Extract the inner zip
                with zipfile.ZipFile(extract_p / inner_name) as inner:
                    inner.extractall(extract_p)
                logger.info("Extracted %s → %s", inner_name, inner_path)
                paths[role] = str(inner_path)
            except Exception as exc:
                logger.warning("Failed to extract %s: %s", inner_name, exc)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full psychotherapy dataset curation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--transfer-zip", default=None,
                        help="Path to the TransferNow zip (for model extraction)")
    parser.add_argument("--extract-dir", default="./models",
                        help="Directory to extract models into")
    parser.add_argument("--intermediate-output", default="./intermediate_dataset.parquet",
                        help="Path for the intermediate dataset")
    parser.add_argument("--final-output", default="./curated_psychotherapy_dataset.jsonl",
                        help="Path for the final JSONL dataset")
    parser.add_argument("--target-size", type=int, default=1500,
                        help="Target number of conversations (500-2000)")
    parser.add_argument("--format", default="llama3", choices=["llama3", "chatml", "openai"],
                        help="Output JSONL format")
    parser.add_argument("--skip-transcription", action="store_true",
                        help="Skip the transcription step")
    parser.add_argument("--audio-dir", default="./mock_clinical_data/audio",
                        help="Directory containing local audio files")
    parser.add_argument("--local-data-dir", default="./mock_clinical_data",
                        help="Directory containing local mock clinical data")
    parser.add_argument("--no-scrub", action="store_true", help="Skip PII scrubbing")
    parser.add_argument("--use-hf", action="store_true",
                        help="Use HuggingFace models instead of local extracted models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without model loading or file output (quick test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # ── Step 0: Extract models from transfer zip ─────────────────────────────
    model_paths = {}
    if not args.use_hf:
        logger.info("Step 0: Extracting models from transfer archive...")
        transfer_zips = list(Path(".").glob("TransferNow-*.zip"))
        if transfer_zips:
            model_paths = extract_models(str(transfer_zips[0]), args.extract_dir)
            logger.info("Model paths: %s", model_paths)
        else:
            logger.warning("No TransferNow zip found — will rely on pre-existing extracted models or HF")

    # ── Step 1: Transcribe local audio (optional) ─────────────────────────────
    if not args.skip_transcription:
        audio_dir = Path(args.audio_dir)
        transcript_dir = Path(args.local_data_dir) / "transcripts"
        if audio_dir.exists() and any(audio_dir.rglob("*.wav")):
            logger.info("Step 1: Transcribing local audio files...")
            try:
                from transcriber import Transcriber
                transcriber = Transcriber(model_name="large-v3", device="auto")
                transcriber.transcribe_directory(
                    str(audio_dir), str(transcript_dir),
                    output_format="jsonl", language=None,
                )
            except Exception as exc:
                logger.error("Transcription failed: %s — continuing without audio", exc)
        else:
            logger.info("Step 1: No local audio files found — skipping transcription")

    # ── Step 2: Build dataset ───────────────────────────────────────────────
    logger.info("Step 2: Building unified dataset...")
    from dataset_builder import build_dataset

    intermediate_df = build_dataset(
        output_path=args.intermediate_output,
        target_size=args.target_size,
        scrub_pii=not args.no_scrub,
        local_data_dir=args.local_data_dir,
        extracted_data_dir=args.extract_dir,
        seed=args.seed,
    )

    # ── Step 3 & 4: Annotate and export ─────────────────────────────────────
    logger.info("Step 3-4: Annotating and exporting...")
    from annotator_and_exporter import AnnotateAndExport

    # Resolve model paths
    emo_11 = model_paths.get("emotion_model_1", "") + "/model"
    emo_7 = model_paths.get("emotion_model", "") + "/models"
    beh = model_paths.get("behaviour_model", "") + "/models/distilbert_behaviour_model"

    # Fallback: check if already extracted
    for candidate in ["./models/emotion_model_1/model", "emotion_model_1/model"]:
        if Path(candidate).exists() and not emo_11:
            emo_11 = candidate
    for candidate in ["./models/emotion_model/models", "emotion_model/models"]:
        if Path(candidate).exists() and not emo_7:
            emo_7 = candidate
    for candidate in ["./models/behaviour_model/models/distilbert_behaviour_model",
                      "behaviour_model/models/distilbert_behaviour_model"]:
        if Path(candidate).exists() and not beh:
            beh = candidate

    annotator = AnnotateAndExport(
        emotion_model_dir_11=emo_11 or None,
        emotion_model_dir_7=emo_7 or None,
        behaviour_model_dir=beh or None,
        use_hf=args.use_hf,
    )

    annotator.run(
        input_path=args.intermediate_output,
        output_path=args.final_output,
        target_size=args.target_size,
        format=args.format,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
