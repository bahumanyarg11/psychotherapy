#!/usr/bin/env python3
"""
Module 1: PII Scrubbing (Device-Level Simulation)
===================================================

Removes Personal Identifiable Information (PII) from raw text before it enters
the main psychotherapy data pipeline.

Uses Microsoft Presidio (analyzer + anonymizer) on the spaCy NLP engine for
NER-based detection, plus custom regex Recognizers for medical/patient IDs,
Indian phone numbers, Aadhaar, and other clinical identifiers.

Designed to be lightweight enough for edge deployment on a dedicated mobile
recording device:
  - Uses spaCy's lightweight ``en_core_web_sm`` model (~12 MB)
  - Lazy initialisation — engines created on first use
  - Batch processing support for throughput on mobile devices
  - No external network calls after model download

Supported entity types
──────────────────────
- PERSON, ORG, GPE, LOCATION, DATE_TIME, EMAIL_ADDRESS, PHONE_NUMBER
- Custom: PATIENT_ID (medical record numbers), MEDICAL_LICENSE,
  INDIAN_AADHAAR, INDIAN_PHONE, URL, IP_ADDRESS

Usage
─────
  python pii_scrubber.py --input "John Smith was seen on Jan 5, 2020."
  python pii_scrubber.py --input-file transcript.txt --output clean.txt
  python pii_scrubber.py --batch conversations.jsonl --format jsonl
  python pii_scrubber.py --test   # run built-in self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# Prevent TensorFlow/Keras initialisation issues during Presidio/Transformers imports
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["KERAS_HOME"] = str(Path(__file__).resolve().parent / ".keras")


from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    Pattern,
    PatternRecognizer,
)
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PII.Scrubber")

# ── Constants ───────────────────────────────────────────────────────────────

# Default spaCy model for edge deployment
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# Entities Presidio detects natively, plus their anonymisation strategy.
# "replace" → masked with a placeholder token.
PII_ENTITY_MAP: dict[str, dict[str, str]] = {
    "PERSON":           {"operator": "replace", "replace_value": "[PERSON]"},
    "LOCATION":         {"operator": "replace", "replace_value": "[LOCATION]"},
    "GPE":              {"operator": "replace", "replace_value": "[LOCATION]"},
    "DATE_TIME":        {"operator": "replace", "replace_value": "[DATE]"},
    "EMAIL_ADDRESS":    {"operator": "replace", "replace_value": "[EMAIL]"},
    "PHONE_NUMBER":     {"operator": "replace", "replace_value": "[PHONE]"},
    "IDENTITY_NO":      {"operator": "replace", "replace_value": "[ID]"},
    "ORGANIZATION":     {"operator": "replace", "replace_value": "[ORG]"},
    "MEDICAL_LICENSE":  {"operator": "replace", "replace_value": "[MEDICAL_ID]"},
    "URL":              {"operator": "replace", "replace_value": "[URL]"},
    "IP_ADDRESS":       {"operator": "replace", "replace_value": "[IP]"},
    "UK_NHS":           {"operator": "replace", "replace_value": "[NHS]"},
    "US_SSN":           {"operator": "replace", "replace_value": "[SSN]"},
    # Custom recognisers registered below
    "PATIENT_ID":       {"operator": "replace", "replace_value": "[PATIENT_ID]"},
    "INDIAN_AADHAAR":   {"operator": "replace", "replace_value": "[AADHAAR]"},
    "INDIAN_PHONE":     {"operator": "replace", "replace_value": "[PHONE]"},
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.55


# ── Custom Recognisers ────────────────────────────────────────────────────────


class MedicalIDRecognizer(PatternRecognizer):
    """Detect medical record numbers and patient IDs via regex.

    Matches patterns like ``MRN: ABC-12345``, ``PAT-2024-001``,
    ``PID 12345``, ``Patient ID: P9876``.
    """

    def __init__(self, name: str = "MedicalIDRecognizer") -> None:
        patterns = [
            Pattern("Patient ID",
                    r"\b(PAT|PID|MRN|Patient ID)[\s:\-]*[A-Z0-9\-]{3,20}\b", 0.90),
            Pattern("SSN-like MRN",
                    r"\b\d{3}-\d{2}-\d{4}\b", 0.85),
        ]
        super().__init__(
            supported_entity="PATIENT_ID",
            name=name,
            patterns=patterns,
            supported_language="en",
        )


class AadhaarNumberRecognizer(PatternRecognizer):
    """Detect Indian Aadhaar numbers (12-digit, grouped as XXXX-XXXX-XXXX)."""

    def __init__(self, name: str = "AadhaarNumberRecognizer") -> None:
        patterns = [
            Pattern("Aadhaar",
                    r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", 0.80),
        ]
        super().__init__(
            supported_entity="INDIAN_AADHAAR",
            name=name,
            patterns=patterns,
            supported_language="en",
        )


class IndianPhoneNumberRecognizer(PatternRecognizer):
    """Detect Indian phone numbers (10 digits, optionally prefixed with +91)."""

    def __init__(self, name: str = "IndianPhoneNumberRecognizer") -> None:
        patterns = [
            Pattern("Indian Phone",
                    r"\b(?:\+91[\s\-]?)?[6-9]\d{9}\b", 0.75),
        ]
        super().__init__(
            supported_entity="INDIAN_PHONE",
            name=name,
            patterns=patterns,
            supported_language="en",
        )


# ── Helper: build RecognizerResult without match_text ────────────────────────


def _make_result(entity_type: str, start: int, end: int, score: float, text: str) -> RecognizerResult:
    """Create a RecognizerResult (this Presidio version has no match_text kwarg)."""
    return RecognizerResult(
        entity_type=entity_type,
        start=start,
        end=end,
        score=score,
        analysis_explanation=None,
        recognition_metadata={
            "match_text": text[start:end] if 0 <= start < len(text) and end <= len(text) else "",
            "recognizer_name": "custom",
        },
    )


# ──── ScrubbedText container ─────────────────────────────────────────────────


class ScrubbedText:
    """Container for scrubbed text and audit metadata."""

    def __init__(self, text: str, original_length: int,
                 entities_found: list[dict], num_masks: int) -> None:
        self.text = text
        self.original_length = original_length
        self.entities_found = entities_found
        self.num_masks = num_masks

    def to_dict(self) -> dict[str, Any]:
        return {
            "scrubbed_text": self.text,
            "original_length": self.original_length,
            "entities_found": self.entities_found,
            "num_masks": self.num_masks,
        }


# ──── PIIScrubber ────────────────────────────────────────────────────────────


class PIIScrubber:
    """
    Lightweight PII scrubber using Microsoft Presidio.

    Optimised for edge deployment:
      - Uses spaCy's lightweight ``en_core_web_sm`` model
      - Lazy initialisation — engines created on first use
      - Batch processing support for throughput on mobile devices

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence score for entity detection (default 0.55).
    add_custom_recognizers : bool
        Whether to register custom regex Recognizers for medical IDs,
        Aadhaar numbers, and Indian phone numbers.
    language : str
        Language code for the NLP engine (default "en").
    spacy_model : str
        Name of the spaCy model to use (default "en_core_web_sm").
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        add_custom_recognizers: bool = True,
        language: str = "en",
        spacy_model: str = DEFAULT_SPACY_MODEL,
    ) -> None:
        self.language = language
        self.confidence_threshold = confidence_threshold
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None
        self._custom_recognizers = add_custom_recognizers
        self._spacy_model = spacy_model

    @property
    def analyzer(self) -> AnalyzerEngine:
        """Lazy-load the Presidio AnalyzerEngine."""
        if self._analyzer is None:
            nlp_engine = SpacyNlpEngine(
                models=[{"lang_code": "en", "model_name": self._spacy_model}]
            )
            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["en"],
            )
            logger.info("PII Analyzer engine initialised (spaCy %s).", self._spacy_model)

            if self._custom_recognizers:
                # Custom regex patterns are applied separately in scrub()
                # to avoid overlap conflicts between Presidio built-in and
                # custom recognizers (e.g. Aadhaar 1234 5678 9012 → DATE_TIME).
                logger.info("Custom regex patterns for clinical PII will be "
                            "applied during scrub().")

        return self._analyzer

    @property
    def anonymizer(self) -> AnonymizerEngine:
        """Lazy-load the Presidio AnonymizerEngine."""
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    # ── Operator config builder ─────────────────────────────────────────────

    @staticmethod
    def _build_operator_configs() -> dict[str, OperatorConfig]:
        """Build the operator config dict for presidio-anonymizer from PII_ENTITY_MAP."""
        operators: dict[str, OperatorConfig] = {}
        for entity_type, cfg in PII_ENTITY_MAP.items():
            operators[entity_type] = OperatorConfig(
                operator_name=cfg["operator"],
                params={"new_value": cfg["replace_value"]},
            )
        return operators

    # ── Core scrub methods ────────────────────────────────────────────────────

    def scrub(self, text: str) -> ScrubbedText:
        """
        Scrub PII from a single text string.

        Returns
        -------
        ScrubbedText
            Container with ``.text`` (scrubbed), ``.entities_found``
            (audit list), and ``.num_masks`` (count).
        """
        if not text or not text.strip():
            return ScrubbedText(text or "", 0, [], 0)

        original_length = len(text)

        # ── Step 1: Presidio NER detection ───────────────────────────────────
        try:
            presidio_entities = self.analyzer.analyze(
                text=text,
                language=self.language,
                score_threshold=self.confidence_threshold,
            )
        except Exception as exc:
            logger.error("Presidio analyze failed on text: %s — %s", text[:80], exc)
            presidio_entities = []

        # ── Step 2: Custom regex patterns (medical IDs, phones, emails) ───────
        custom_entities: list[RecognizerResult] = []
        for pattern_str, label in self._extra_regex_patterns():
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                # Skip if already covered by a Presidio entity of the SAME type
                same_type_overlap = any(
                    e.entity_type == label and
                    (e.start <= match.start() < e.end or
                     e.start < match.end() <= e.end)
                    for e in presidio_entities
                )
                if not same_type_overlap:
                    custom_entities.append(
                        _make_result(label, match.start(), match.end(), 0.80, text)
                    )

        all_entities = list(presidio_entities) + custom_entities

        # Give custom regex entities priority over Presidio entities on overlap
        # (e.g. Aadhaar 1234 5678 9012 looks like a date to spaCy)
        filtered_presidio = []
        for p in presidio_entities:
            overlaps_custom = any(
                c.start <= p.start < c.end or c.start < p.end <= c.end
                for c in custom_entities
            )
            if not overlaps_custom:
                filtered_presidio.append(p)
        all_entities = filtered_presidio + custom_entities

        # ── Step 3: Build audit log ──────────────────────────────────────────
        entities_found = [
            {
                "type": e.entity_type,
                "text": text[e.start:e.end] if 0 <= e.start < len(text) and e.end <= len(text) else "",
                "confidence": round(e.score, 3),
                "position": [e.start, e.end],
            }
            for e in sorted(all_entities, key=lambda x: x.start)
        ]

        # ── Step 4: Anonymise ────────────────────────────────────────────────
        operator_configs = self._build_operator_configs()
        try:
            scrubbed = self.anonymizer.anonymize(
                text=text,
                analyzer_results=all_entities,
                operators=operator_configs,
            )
            scrubbed_text = scrubbed.text
        except Exception as exc:
            logger.warning("Presidio anonymizer failed (%s); falling back to manual masking.", exc)
            scrubbed_text = self._manual_mask(text, all_entities)

        return ScrubbedText(
            text=scrubbed_text,
            original_length=original_length,
            entities_found=entities_found,
            num_masks=len(all_entities),
        )

    def scrub_batch(self, texts: list[str]) -> list[ScrubbedText]:
        """Batch-scrub a list of text strings."""
        results: list[ScrubbedText] = []
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                logger.info("Scrubbing batch: %d/%d", i, len(texts))
            results.append(self.scrub(text))
        return results

    def scrub_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        format: str = "txt",
    ) -> str:
        """
        Scrub PII from a file.

        Supports:
          - ``txt``  — one utterance/conversation per line
          - ``jsonl`` — one JSON object per line with a ``text`` field
          - ``json``  — list of dicts with ``text`` key, or dict of {id: text}
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = str(input_path.parent / f"{input_path.stem}.scrubbed{input_path.suffix}")
        output_path = Path(output_path)
        logger.info("Scrubbing file: %s -> %s", input_path, output_path)

        scrub_audit: list[dict] = []
        total_entities = 0

        with open(input_path, "r", encoding="utf-8") as f:
            if format == "txt":
                lines = f.readlines()
                scrubbed_lines = []
                for i, line in enumerate(lines):
                    result = self.scrub(line.strip())
                    scrubbed_lines.append(result.text)
                    if result.entities_found:
                        scrub_audit.append({"line": i, "entities": result.entities_found})
                        total_entities += result.num_masks
                output_path.write_text("\n".join(scrubbed_lines), encoding="utf-8")

            elif format == "jsonl":
                scrubbed_records = []
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict) and "text" in record:
                        result = self.scrub(record["text"])
                        record["text"] = result.text
                        if result.entities_found:
                            scrub_audit.append({"line": i, "entities": result.entities_found})
                            total_entities += result.num_masks
                    scrubbed_records.append(record)
                with open(output_path, "w", encoding="utf-8") as out:
                    for rec in scrubbed_records:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elif format == "json":
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            result = self.scrub(item["text"])
                            item["text"] = result.text
                            if result.entities_found:
                                total_entities += result.num_masks
                elif isinstance(data, dict):
                    for key in data:
                        if isinstance(data[key], str):
                            result = self.scrub(data[key])
                            data[key] = result.text
                            total_entities += result.num_masks
                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(data, out, ensure_ascii=False, indent=2)

        logger.info("Scrubbing complete. %d entities masked in %d instances.",
                     total_entities, len(scrub_audit))

        # Write audit file
        audit_path = output_path.parent / f"{output_path.stem}_pii_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({"audit": scrub_audit, "total_entities": total_entities}, f, indent=2)
        logger.info("PII audit log written to: %s", audit_path)
        return str(output_path)

    # ── Compatibility aliases ────────────────────────────────────────────────

    def scrub_text(self, text: str) -> tuple[str, list[dict]]:
        """Alias for ``scrub`` — returns ``(cleaned_text, pii_audit_list)``.

        Provided for backwards compatibility with callers that expect a
        two-element tuple rather than a ``ScrubbedText`` object.
        """
        result = self.scrub(text)
        return result.text, result.entities_found

    def scrub_text_batch(self, texts: list[str]) -> tuple[list[str], list[list[dict]]]:
        """Alias for ``scrub_batch`` — returns ``(cleaned_texts, pii_audits)``.

        Provided for backwards compatibility.
        """
        results = self.scrub_batch(texts)
        return [r.text for r in results], [r.entities_found for r in results]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extra_regex_patterns() -> list[tuple[str, str]]:
        """Additional regex patterns for clinical PII not covered by Presidio."""
        return [
            # Patient IDs: MRN: ABC-12345, PID-2024-001, Patient ID: P9876 (alphanumeric, ≥3 chars)
            (r"\b(?:MRN|PID|Patient\s*ID)[\s:\-]*[A-Z0-9][A-Z0-9\-]{2,19}\b", "PATIENT_ID"),
            # SSN-like medical record numbers: 123-45-6789
            (r"\b\d{3}-\d{2}-\d{4}\b", "PATIENT_ID"),
            # Indian Aadhaar: 1234 5678 9012 or 1234-5678-9012
            (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "INDIAN_AADHAAR"),
            # US phone numbers: (555) 123-4567, 555-123-4567
            (r"\(\d{3}\)\s*\d{3}[\s\-]?\d{4}\b", "PHONE_NUMBER"),
            # Indian phone: +91-9876543210, 9876543210
            (r"(?:\+91[\s\-]?)?[6-9]\d{9}\b", "INDIAN_PHONE"),
            # Emails
            (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "EMAIL_ADDRESS"),
        ]

    def _manual_mask(self, text: str, entities: list[RecognizerResult]) -> str:
        """Fallback manual masking when the anonymizer fails."""
        if not entities:
            return text
        sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)
        result = text
        for e in sorted_entities:
            mask = PII_ENTITY_MAP.get(e.entity_type, {}).get("replace_value", "[MASK]")
            result = result[:e.start] + mask + result[e.end:]
        return result


# ──── Self-test ────────────────────────────────────────────────────────────────


def self_test() -> bool:
    """Run built-in self-tests to verify PII scrubbing."""
    test_cases = [
        {
            "input": "Patient John Smith was admitted on Jan 5, 2020 with MRN: ABC-12345.",
            "expect_masks": ["[PERSON]", "[DATE]", "[PATIENT_ID]"],
        },
        {
            "input": "Call me at (555) 123-4567 or email john.smith@example.com.",
            "expect_masks": ["[PHONE]", "[EMAIL]"],
        },
        {
            "input": "Dr. Jane Doe from Harvard University prescribed 10mg medication.",
            "expect_masks": ["[PERSON]", "[ORG]"],
        },
        {
            "input": "The patient mentioned living in Boston, Massachusetts.",
            "expect_masks": ["[LOCATION]"],
        },
        {
            "input": "Aadhaar: 1234 5678 9012, phone +91-9876543210.",
            "expect_masks": ["[AADHAAR]", "[PHONE]"],
        },
        {
            "input": "No personal information here — just a general statement about therapy.",
            "expect_masks": [],
        },
    ]

    print("=" * 60)
    print("PII Scrubber Self-Test")
    print("=" * 60)

    scrubber = PIIScrubber()
    all_passed = True

    for i, tc in enumerate(test_cases, 1):
        result = scrubber.scrub(tc["input"])
        missing = [m for m in tc["expect_masks"] if m not in result.text]
        passed = len(missing) == 0
        status = "PASS" if passed else "FAIL"

        print(f"\nTest {i}: [{status}]")
        print(f"  Input : {tc['input']}")
        print(f"  Output: {result.text}")
        print(f"  Masks: {result.num_masks} entities → {[e['type'] for e in result.entities_found]}")
        if missing:
            print(f"  Missing: {missing}")
            all_passed = False

    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed


# ──── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PII Scrubber — detect and mask personal information in text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pii_scrubber.py --test
  python pii_scrubber.py --input "John Smith went to Boston on March 3, 2021."
  python pii_scrubber.py --input-file transcript.txt --output clean_transcript.txt
  python pii_scrubber.py --batch conversations.jsonl --format jsonl
        """,
    )
    parser.add_argument("--input", "-i", type=str, help="Single text string to scrub.")
    parser.add_argument("--input-file", "-if", type=str, help="Path to input file (txt/json/jsonl).")
    parser.add_argument("--output", "-o", type=str, help="Output file path (for file mode).")
    parser.add_argument("--batch", "-b", type=str, help="Path to batch file (jsonl with 'text' field).")
    parser.add_argument("--format", "-f", type=str, default="txt",
                        choices=["txt", "json", "jsonl"], help="File format for input file (default: txt).")
    parser.add_argument("--test", action="store_true", help="Run built-in self-test.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold for PII detection (default: {DEFAULT_CONFIDENCE_THRESHOLD}).")
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if self_test() else 1)

    if args.input:
        scrubber = PIIScrubber(confidence_threshold=args.threshold)
        result = scrubber.scrub(args.input)
        print(f"Original : {args.input}")
        print(f"Scrubbed : {result.text}")
        print(f"Entities : {len(result.entities_found)} | Masks: {result.num_masks}")
        for e in result.entities_found:
            print(f"  [{e['type']}] '{e['text']}' (conf={e['confidence']})")

    elif args.input_file:
        scrubber = PIIScrubber(confidence_threshold=args.threshold)
        output = scrubber.scrub_file(args.input_file, args.output, format=args.format)
        print(f"Output written to: {output}")

    elif args.batch:
        scrubber = PIIScrubber(confidence_threshold=args.threshold)
        records = []
        with open(args.batch, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        texts = [r.get("text", r.get("utterance", "")) for r in records]
        results = scrubber.scrub_batch(texts)
        for rec, res in zip(records, results):
            key = "text" if "text" in rec else "utterance"
            rec[key] = res.text
        out_path = args.batch.replace(".jsonl", ".scrubbed.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Batch scrubbed: {len(records)} records -> {out_path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
