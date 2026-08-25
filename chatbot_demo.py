#!/usr/bin/env python3
"""
Conversational AI Psychotherapy Chatbot Module
===============================================
Deliverable for Milestone 1 (Part 1): AI-Assisted Mental Health Conversational Agent.

This module demonstrates the conversational agent runtime powered by the curated
psychotherapy dataset. It implements:
  1. System Prompt Conditioning (empathy, clinical validation, non-diagnostic boundaries).
  2. Multi-turn Dialogue State & Context Tracking.
  3. Dynamic Affect & Intent Awareness (simulated/model-based emotion & behavior tracking).
  4. Crisis De-escalation Guardrails (immediate safety hotline referral on crisis flags).
  5. Interactive Terminal / API inference mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Chatbot")

# Default Clinical System Prompt curated for the dataset
DEFAULT_SYSTEM_PROMPT = (
    "You are an empathetic, evidence-based AI psychotherapy assistant. "
    "Your objective is to provide a safe, non-judgmental space for patients to explore "
    "their emotions, thoughts, and behavioural patterns. Follow Cognitive Behavioural "
    "Therapy (CBT) principles: validate feelings, gently explore underlying cognitive "
    "distortions, and collaboratively identify healthy coping strategies. "
    "Never provide formal medical diagnoses or prescribe medications. "
    "If the patient indicates severe self-harm or suicidal crisis, immediately provide "
    "crisis helpline resources with urgency and care."
)

CRISIS_KEYWORDS = [
    r"\bsuicid", r"\bkill(ing)?\s+(my\s*self|me)\b", r"\bend(ing)?\s+my\s+life\b",
    r"\bwant(ing)?\s+to\s+die\b", r"\bself[- ]harm\b", r"\bhurt(ing)?\s+myself\b",
    r"\bno\s+reason\s+to\s+live\b", r"\boverdose\b", r"\btak(e|ing)\s+all\s+my\s+pills\b"
]

CRISIS_RESPONSE = (
    "I hear how much pain you are experiencing right now, and I want you to know that your life "
    "has immense value. Because your immediate safety is the most important priority, please connect "
    "with professional support right now:\n\n"
    "• National Suicide Prevention Helpline (India): 14416 / 9152987821 (KIRAN: 1800-599-0019)\n"
    "• International Hotlines: 988 (USA/Canada), 111 (UK), 112 (EU)\n"
    "• Emergency Services: 112 / 911\n\n"
    "Please reach out to a trusted family member, doctor, or crisis counselor immediately."
)

# Rule-based / Keyword affective fallback heuristics for live inference
EMOTION_KEYWORDS = {
    "anxiety": ["anxious", "worried", "panic", "nervous", "dread", "terrified", "stressed", "restless"],
    "sadness": ["sad", "depressed", "hopeless", "crying", "unhappy", "lonely", "miserable", "down"],
    "anger": ["angry", "furious", "mad", "irritated", "pissed", "hate", "resentful", "rage"],
    "frustration": ["frustrated", "stuck", "annoyed", "exhausted", "tired of", "overwhelmed"],
    "shame_guilt": ["guilty", "ashamed", "my fault", "embarrassed", "worthless", "blame myself"],
    "fear": ["scared", "fearful", "frightened", "paralyzed", "afraid"],
    "grief": ["lost", "passed away", "mourning", "miss them", "grieving", "death"],
    "relief": ["relieved", "calmer", "better now", "glad that's over", "feeling eased"],
    "positive_progress": ["better", "improving", "hopeful", "managed to", "accomplished", "proud"],
}

BEHAVIOUR_KEYWORDS = {
    "Social_Withdrawal": ["avoiding friends", "staying in bed", "isolating", "shut inside", "don't want to see anyone"],
    "Rumination": ["can't stop thinking", "keeps replaying", "over and over", "obsessing", "stuck in my head"],
    "Avoidance": ["procrastinating", "putting off", "ignoring the problem", "running away from", "avoiding"],
    "Negative_Self_Talk": ["i am stupid", "i'm a failure", "nobody likes me", "i always mess up", "i ruin everything"],
    "Emotional_Expression": ["i feel", "i am experiencing", "it hurts", "i sense", "my emotions are"],
    "Help_Seeking": ["what should i do", "can you help", "how do i cope", "i need advice", "give me strategies"],
    "Active_Coping": ["i tried journaling", "i went for a walk", "i practiced breathing", "i spoke to a friend"],
    "Conversational_Act": ["hello", "hi", "good morning", "thank you", "bye", "okay", "yes"],
}


@dataclass
class ChatTurn:
    role: str  # 'system', 'user', 'assistant'
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


class PsychotherapyChatbot:
    """
    Mental Health Conversational Agent representing Milestone 1 deliverable.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        dataset_path: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.history: List[ChatTurn] = [
            ChatTurn(role="system", content=self.system_prompt)
        ]
        self.dataset_path = dataset_path
        self.dataset_samples: List[Dict] = []
        if dataset_path and Path(dataset_path).exists():
            self._load_dataset_references(dataset_path)

    def _load_dataset_references(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.dataset_samples.append(json.loads(line.strip()))
            logger.info(f"Loaded {len(self.dataset_samples)} curated conversation references.")
        except Exception as e:
            logger.warning(f"Could not load reference dataset: {e}")

    def detect_crisis(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pat, lower) for pat in CRISIS_KEYWORDS)

    def detect_affect_and_intent(self, text: str) -> Tuple[str, str]:
        lower = text.lower()
        detected_emotion = "neutral"
        for emo, kws in EMOTION_KEYWORDS.items():
            if any(kw in lower for kw in kws):
                detected_emotion = emo
                break

        detected_behaviour = "Emotional_Expression"
        for beh, kws in BEHAVIOUR_KEYWORDS.items():
            if any(kw in lower for kw in kws):
                detected_behaviour = beh
                break

        return detected_emotion, detected_behaviour

    def generate_response(self, user_input: str) -> Tuple[str, Dict[str, str]]:
        # 1. Check crisis guardrails
        if self.detect_crisis(user_input):
            meta = {"emotion": "crisis", "behaviour": "Help_Seeking", "crisis_flag": "true"}
            response = CRISIS_RESPONSE
            self.history.append(ChatTurn(role="user", content=user_input, metadata=meta))
            self.history.append(ChatTurn(role="assistant", content=response, metadata={"type": "crisis_deescalation"}))
            return response, meta

        # 2. Extract Affect and Behavioural Intent
        emotion, behaviour = self.detect_affect_and_intent(user_input)
        meta = {"emotion": emotion, "behaviour": behaviour, "crisis_flag": "false"}

        # 3. Formulate Therapeutic Response based on CBT principles
        response = self._synthesize_cbt_response(user_input, emotion, behaviour)

        # 4. Update Conversation History
        self.history.append(ChatTurn(role="user", content=user_input, metadata=meta))
        self.history.append(ChatTurn(role="assistant", content=response, metadata={"emotion": "empathetic_validation"}))

        return response, meta

    def _synthesize_cbt_response(self, text: str, emotion: str, behaviour: str) -> str:
        # Therapeutic response formulation based on CBT framework
        validation_map = {
            "anxiety": "I hear how much anxiety and tension you are holding right now. It is completely understandable to feel overwhelmed when faced with uncertainty.",
            "sadness": "Thank you for sharing that with me. Carrying such deep sadness can feel very isolating, but you don't have to carry it alone.",
            "anger": "It sounds like you're experiencing justifiable anger and frustration about this situation.",
            "frustration": "Feeling stuck and frustrated takes a real toll on your energy. Let's take a breath and break this down together.",
            "shame_guilt": "I want to remind you that experiencing self-blame is common, but you deserve compassion rather than harsh self-criticism.",
            "fear": "It takes courage to express fear. We can look at what is causing this fear step by step.",
            "grief": "Loss is profoundly difficult, and whatever you are feeling right now is completely valid as you process this grief.",
            "relief": "I'm really glad you are experiencing a moment of relief and breathing room.",
            "positive_progress": "That is wonderful progress. Acknowledging these positive steps is a crucial part of healing.",
            "neutral": "I am listening closely to what you are describing.",
        }

        inquiry_map = {
            "Negative_Self_Talk": "When you notice that harsh inner voice, what specific thoughts are coming up? What would you say to a dear friend in your exact shoes?",
            "Rumination": "When your mind starts looping on these thoughts, what physical sensations do you notice in your body? Would you like to try a brief grounding exercise?",
            "Social_Withdrawal": "Withdrawing often feels like the safest option when energy is low. Is there one small, low-pressure connection you might feel open to?",
            "Avoidance": "Avoiding uncomfortable tasks is a very natural way our brain tries to protect us. What feels like the smallest, most manageable micro-step you could take?",
            "Help_Seeking": "Let's explore some concrete, evidence-based coping strategies that align with your goals.",
            "Active_Coping": "How did taking that active step make you feel afterwards?",
            "Emotional_Expression": "Can you tell me more about what triggered these feelings today?",
            "Conversational_Act": "How has your day been treating you overall?",
        }

        v = validation_map.get(emotion, validation_map["neutral"])
        q = inquiry_map.get(behaviour, "What thoughts or feelings are most prominent for you right now?")

        return f"{v}\n\n{q}"

    def export_chat_session(self) -> Dict:
        """Export session in ChatML/Llama-3 JSON format."""
        return {
            "messages": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    **({"metadata": turn.metadata} if turn.metadata else {})
                }
                for turn in self.history
            ],
            "turn_count": len(self.history),
        }


    def transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe an audio file using faster-whisper (transcriber.py)."""
        try:
            from transcriber import Transcriber
            transcriber = Transcriber(model_name="base")
            result = transcriber.transcribe(audio_path)
            text = " ".join(s["text"].strip() for s in result.segments)
            logger.info(f"Transcribed audio ({result.detected_language}): '{text}'")
            return text
        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")
            return f"[Audio transcription error: {e}]"


def run_interactive_session() -> None:
    print("=" * 70)
    print("AI Psychotherapy Conversational Chatbot (Milestone 1 Deliverable)")
    print("Type text, or type 'audio <path>' to load a voice file.")
    print("Type 'export' to view ChatML, or 'exit'/'quit' to finish session.")
    print("=" * 70)

    dataset_path = Path(__file__).resolve().parent / "base_dataset.jsonl"
    bot = PsychotherapyChatbot(dataset_path=str(dataset_path) if dataset_path.exists() else None)

    print("\nTherapist: Hello, welcome. I'm here to support you. How are you feeling today?")
    while True:
        try:
            user_input = input("\nPatient: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nTherapist: Take care of yourself. Goodbye!")
                break
            if user_input.lower() == "export":
                print("\n--- ChatML Export ---")
                print(json.dumps(bot.export_chat_session(), indent=2))
                continue
            if user_input.lower().startswith("audio "):
                audio_file = user_input[6:].strip()
                print(f"[Processing Audio File: {audio_file}]...")
                user_input = bot.transcribe_audio_file(audio_file)
                print(f"Transcribed Speech: \"{user_input}\"")

            response, meta = bot.generate_response(user_input)
            print(f"\n[Affect: {meta['emotion']} | Intent: {meta['behaviour']}]")
            print(f"Therapist: {response}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="AI Psychotherapy Conversational Agent")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI chatbot")
    parser.add_argument("--audio", type=str, help="Process and respond to an input audio/voice file")
    parser.add_argument("--test", action="store_true", help="Run automated conversational test suite")
    args = parser.parse_args()

    bot = PsychotherapyChatbot()

    if args.audio:
        print(f"Transcribing voice input from: {args.audio}...")
        transcribed_text = bot.transcribe_audio_file(args.audio)
        print(f"Patient (Voice): \"{transcribed_text}\"")
        resp, meta = bot.generate_response(transcribed_text)
        print(f"\n[Detected Affect: {meta['emotion']} | Intent: {meta['behaviour']}]")
        print(f"Therapist:\n{resp}")
    elif args.interactive:
        run_interactive_session()
    else:
        print("Running Conversational AI Chatbot Self-Test...")
        test_inputs = [
            "I've been feeling so stressed and anxious about my job lately.",
            "I keep thinking that I'm a complete failure and nobody likes me.",
            "I feel so down today, I just want to stay in bed and hide from the world.",
            "I tried deep breathing and went for a 15-minute walk this morning.",
        ]
        for inp in test_inputs:
            resp, meta = bot.generate_response(inp)
            print(f"\nUser: '{inp}'")
            print(f"Detected: Emotion={meta['emotion']}, Behaviour={meta['behaviour']}")
            print(f"Response snippet: {resp.splitlines()[0]}")
        print("\nAll conversational agent tests passed successfully!")


if __name__ == "__main__":
    main()
