# Psychotherapy Dataset Curation Pipeline

A modular Python codebase for preparing and curating a psychotherapy dataset
comprising emotional, behavioural, contextual, and therapeutic interaction
data for training and evaluating AI-assisted mental health models.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Extract pre-trained models from the transfer zip
python run_pipeline.py            # auto-detects TransferNow-*.zip

# 3. Run the full pipeline (end-to-end)
python run_pipeline.py --target-size 1500

# 4. Or run modules individually
python dataset_builder.py --output ./intermediate.parquet
python annotator_and_exporter.py --input ./intermediate.parquet \
    --output ./curated_psychotherapy_dataset.jsonl \
    --target-size 1500
```

## Architecture

```
                           ┌─────────────────┐
                           │  Local Mock     │
                           │  Audio (.wav)   │
                           └────────┬────────┘
                                    │ (Whisper)
                                    ▼
                           ┌─────────────────┐
                           │  transcriber.py │
                           │  (Whisper ASR)  │
                           └────────┬────────┘
                                    │ JSON transcripts
                                    ▼
┌──────────┐  ┌──────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│HF Datasets│ │Local Mock Data│   │  dataset_builder.py   │   │  pii_scrubber.py│
│(GoEmotions│ │(text + audio) │   │  (aggregation &       │◄──(presidio scrub)
│EmoDialog  │ │              │   │   normalisation)      │   └─────────────────┘
│CounselChat│ │Extracted Zips│   └─────────┬────────────┘
│MedDialog  │ │              │             │
└──────────┘ └──────────────┘             │ intermediate.parquet
                                         ▼
                              ┌──────────────────────────┐
                              │ annotator_and_exporter.py│
                              │  (emotion + intent labels)│
                              └─────────┬────────────────┘
                                        │
                                        ▼
                              ┌──────────────────────────┐
                              │ curated_psychotherapy_   │
                              │ dataset.jsonl (Llama-3)  │
                              └──────────────────────────┘
```

## Modules

### 1. `pii_scrubber.py` — PII Removal (Device-Level)

- Uses Microsoft Presidio (analyzer + anonymizer) on the spaCy NLP engine
- Custom recognizers for medical IDs, Indian Aadhaar numbers, Indian phone numbers
- Edge-deployable: ~12 MB spaCy model footprint
- Auditable: produces a PII mapping file for clinician review

```bash
python pii_scrubber.py --input raw.txt --output clean.txt --audit audit.jsonl
```

### 2. `transcriber.py` — Multilingual Audio-to-Text

- Uses `faster-whisper` (Rust-accelerated, supports MPS/CPU)
- Code-switching support: multilingual models (large-v3 recommended) auto-detect
  mixed-language segments
- Silero VAD filtering to reduce hallucination
- Supports .wav, .mp3, .m4a, .flac, .ogg, .mp4, .webm

```bash
python transcriber.py --input-dir ./audio/ --output-dir ./transcripts/ --model large-v3
```

### 3. `dataset_builder.py` — Data Aggregation & Annotation Prep

Fetches and normalizes data from:

| Source | Type | Notes |
|--------|------|-------|
| GoEmotions | Emotion labels | 27 fine-grained emotions, mapped to 11-class taxonomy |
| EmpatheticDialogues | Conversational | Context + empathetic response |
| CounselChat | Therapeutic Q&A | Question = patient, answer = therapist |
| DailyDialog | Conversational | Turn-by-turn dialogues with emotion labels |
| Mental Health FAQ | Q&A | Medical questions with categories |
| MedDialog | Medical Q&A | Doctor-patient conversations |
| RAVDESS / CREMA-D | Audio + labels | Speech emotion datasets (metadata only) |
| Local mock data | Text / audio | Patient session simulations (EN + 5 Indian langs) |
| Extracted zips | Pre-modelled data | Cleaned emotion & behaviour datasets |

Quality filters:
1. Drop utterances < 5 words
2. Drop conversations < 2 turns (need patient + therapist)
3. Drop conversations with no patient utterances
4. Deduplicate exact-text utterances within each source
5. Stratified down-s / up-sampling to target size (default 1,500)

```bash
python dataset_builder.py --output ./intermediate.parquet --target-size 1500
```

### 4. `annotator_and_exporter.py` — Annotation & Export

**Emotion models** (11-class):
- Primary: `emotion_model_1/model/` — fine-tuned RoBERTa, 11 psychotherapy-emotions
- Fallback: `emotion_model/models/` — 7-class DistilBERT (Ekman)
- Last resort: HuggingFace `j-hartmann/emotion-english-distilroberta-base`

**Behaviour models** (8-class):
- Primary: `behaviour_model/models/distilbert_behaviour_model/` — fine-tuned DistilBERT
- Fallback: Rule-based keyword classifier (same 8 classes)

**Export format** — Llama-3 / ChatML JSONL:

```jsonl
{"id": "counselchat_42", "messages": [
  {"role": "system", "content": "You are a compassionate...".},
  {"role": "user", "content": "[emotion: anxiety | intent: Help_Seeking] I'm so worried..."},
  {"role": "assistant", "content": "It sounds like you're going through a lot..."}
], "metadata": {"source": "CounselChat", "language": "en", "utterance_annotations": [...]}}
```

```bash
python annotator_and_exporter.py --input intermediate.parquet --output dataset.jsonl \
    --emotion-model-dir-11 ./models/emotion_model_1/model \
    --behaviour-model-dir ./models/behaviour_model/models/distilbert_behaviour_model
```

### 5. `run_pipeline.py` — End-to-End Orchestrator

Extracts models from the transfer zip, runs all steps, and produces the final JSONL.

```bash
python run_pipeline.py --target-size 1500 --format llama3
```

## Emotion Taxonomy (11-class)

| # | Label | Description |
|---|-------|-------------|
| 0 | anger | Frustration, irritation, rage |
| 1 | frustration | Blocked goals, irritability |
| 2 | sadness | Grief, disappointment, low mood |
| 3 | confusion | Cognitive disorientation, uncertainty |
| 4 | shame_guilt | Self-reproach, shame, guilt |
| 5 | fear | General fear, terror |
| 6 | anxiety | Worry, nervousness, panic |
| 7 | relief | Easing of distress after resolution |
| 8 | grief | Bereavement, loss processing |
| 9 | positive_progress | Hope, improvement, motivation |
| 10 | neutral | Neutral affect |

## Behaviour Taxonomy (8-class)

| # | Label | Description |
|---|-------|-------------|
| 0 | Social_Withdrawal | Isolating from social contact |
| 1 | Rumination | Repetitive negative thought cycles |
| 2 | Avoidance | Evading tasks, emotions, or problems |
| 3 | Negative_Self_Talk | Self-criticism, self-blame |
| 4 | Emotional_Expression | Direct verbalisation of affect |
| 5 | Help_Seeking | Requesting advice or resources |
| 6 | Active_Coping | Constructive problem-solving actions |
| 7 | Conversational_Act | Greetings, logistics, small talk |

## Output Format

The final `curated_psychotherapy_dataset.jsonl` follows the Llama-3 chat
template, compatible with:

- HuggingFace TRL `SFTTrainer`
- Axolotl fine-tuning
- HuggingFace `transformers` `AutoTokenizer.apply_chat_template()`

Each line is a JSON object with:
- `"messages"`: list of `{"role": "system"|"user"|"assistant", "content": str}`
- `"metadata"`: source, language, turn count, per-utterance emotion/intent annotations

## Presentation GUI

The repository includes a lightweight, local Streamlit interface for presenting
the conversational agent without downloading a large language model:

```bash
pip install streamlit
streamlit run app.py
```

Open the local URL shown in the terminal, then use the suggested prompts or
type your own message. The interface includes session-level affect/intent
signals, a crisis safety response, a new-session control, and JSON export.
This is a demonstration tool, not a licensed therapist or a replacement for
professional mental-health care.
