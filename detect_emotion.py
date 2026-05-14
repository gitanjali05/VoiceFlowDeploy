# pip install openai-whisper transformers torch

from pathlib import Path
import logging
import os
import re

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(BASE_DIR / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("MODELSCOPE_CACHE", str(BASE_DIR / ".cache" / "modelscope"))
os.environ.setdefault(
    "MODELSCOPE_CREDENTIALS_PATH",
    str(BASE_DIR / ".cache" / "modelscope" / "credentials"),
)

import soundfile as sf
import whisper
from funasr import AutoModel
from transformers import pipeline


EMOTIONS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

AUDIO_LABEL_MAP = {
    "angry": "angry",
    "anger": "angry",
    "ang": "angry",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "fearful": "fearful",
    "fear": "fearful",
    "happy": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprised": "surprised",
    "surprise": "surprised",
}

TEXT_LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}

EMOTION2VEC_MODEL_NAME = "emotion2vec_plus_base"
EMOTION2VEC_MODEL_HUB = "hf"
WHISPER_MODEL_SIZE = "base"
TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
OUTPUT_DIR = BASE_DIR / "outputs"
FRUSTRATION_PATTERNS = {
    "angry": [
        r"\bwhat are you doing\b",
        r"\bhow could you\b",
        r"\bhow dare you\b",
        r"\bthis is unacceptable\b",
        r"\bcompletely unacceptable\b",
        r"\bare you serious\b",
        r"\bwhat the\b",
        r"\bwhy would you\b",
        r"\bfix this\b",
        r"\bthis is ridiculous\b",
        r"\bthis is frustrating\b",
        r"\bi am frustrated\b",
        r"\bi'm frustrated\b",
        r"\bi am angry\b",
        r"\bi'm angry\b",
    ]
}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

emotion2vec_model = None
whisper_model = None
text_emotion_classifier = None


# Loads emotion2vec only once, using the smaller base model to avoid a huge download.
def get_emotion2vec_model():
    global emotion2vec_model
    if emotion2vec_model is None:
        emotion2vec_model = AutoModel(
            model=EMOTION2VEC_MODEL_NAME,
            hub=EMOTION2VEC_MODEL_HUB,
            disable_update=True,
        )

    return emotion2vec_model


# Loads Whisper only once, when transcription is first needed.
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    return whisper_model


# Loads the text emotion classifier only once, after Whisper returns text.
def get_text_emotion_classifier():
    global text_emotion_classifier
    if text_emotion_classifier is None:
        text_emotion_classifier = pipeline(
            "text-classification",
            model=TEXT_MODEL_NAME,
            top_k=None,
        )

    return text_emotion_classifier


# Converts labels from a model-specific label map into the shared emotion names.
def normalize_label(label, label_map):
    label_text = str(label).strip().lower().split("/")[-1]
    return label_map.get(label_text)


# Creates an empty score dictionary with every shared emotion set to zero.
def empty_scores():
    return {emotion: 0.0 for emotion in EMOTIONS}


# Measures loudness from the recorded audio so loud frustrated speech can affect the result.
def get_audio_profile(filepath):
    audio, _ = sf.read(str(filepath), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return {
        "rms": float((audio**2).mean() ** 0.5),
        "peak": float(abs(audio).max()) if len(audio) else 0.0,
    }


# Converts emotion2vec labels and scores into the shared seven-emotion score dictionary.
def normalize_audio_scores(result):
    scores_by_emotion = empty_scores()
    item = result[0] if isinstance(result, list) else result
    labels = item.get("labels", [])
    scores = item.get("scores", [])

    if scores and isinstance(scores[0], list):
        scores = scores[0]

    for label, score in zip(labels, scores):
        emotion = normalize_label(label, AUDIO_LABEL_MAP)
        if emotion:
            scores_by_emotion[emotion] = max(scores_by_emotion[emotion], float(score))

    return scores_by_emotion


# Converts text classifier labels and scores into the shared seven-emotion score dictionary.
def normalize_text_scores(result):
    scores_by_emotion = empty_scores()
    rows = result[0] if result and isinstance(result[0], list) else result

    for row in rows:
        emotion = normalize_label(row.get("label", ""), TEXT_LABEL_MAP)
        if emotion:
            scores_by_emotion[emotion] = max(scores_by_emotion[emotion], float(row["score"]))

    return scores_by_emotion


# Runs emotion2vec on an audio file and returns scores for every shared emotion.
def get_audio_scores(filepath):
    audio, sample_rate = sf.read(str(filepath), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = get_emotion2vec_model().generate(
        audio,
        output_dir=str(OUTPUT_DIR),
        fs=sample_rate,
        granularity="utterance",
        extract_embedding=False,
        key=Path(filepath).stem,
    )
    return normalize_audio_scores(result)


# Transcribes the audio file with Whisper and returns the transcript text.
def transcribe_audio(filepath):
    audio, sample_rate = sf.read(str(filepath), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        logging.warning(
            "Whisper expects 16 kHz audio; got %s Hz. Transcription may be poor.",
            sample_rate,
        )

    result = get_whisper_model().transcribe(audio, fp16=False)
    return str(result.get("text", "")).strip()


# Runs text emotion classification on the transcript and returns shared emotion scores.
def get_text_scores(transcript):
    result = get_text_emotion_classifier()(transcript)
    return normalize_text_scores(result)


# Combines audio and text scores so voice tone leads, while words can break close ties.
def fuse_scores(audio_scores, text_scores):
    return {
        emotion: (0.6 * audio_scores.get(emotion, 0.0))
        + (0.4 * text_scores.get(emotion, 0.0))
        for emotion in EMOTIONS
    }


# Boosts anger when the words are confrontational, especially if the voice is loud.
def apply_contextual_adjustments(scores, transcript, audio_profile):
    adjusted = scores.copy()
    text = transcript.lower()
    is_loud = audio_profile["rms"] >= 0.025 or audio_profile["peak"] >= 0.18

    for emotion, patterns in FRUSTRATION_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            boost = 0.28 + (0.14 if is_loud else 0.0)
            adjusted[emotion] = min(1.0, adjusted.get(emotion, 0.0) + boost)
            adjusted["neutral"] = adjusted.get("neutral", 0.0) * 0.55

    return adjusted


# Detects emotion from audio, adds transcript-aware text fusion, and returns the top result.
def detect_emotion(filepath):
    audio_path = Path(filepath)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_scores = get_audio_scores(audio_path)
    audio_profile = get_audio_profile(audio_path)

    try:
        transcript = transcribe_audio(audio_path)
    except Exception as error:
        logging.warning("Whisper transcription failed; using audio-only result: %s", error)
        transcript = ""

    if transcript:
        text_scores = get_text_scores(transcript)
        final_scores = fuse_scores(audio_scores, text_scores)
        final_scores = apply_contextual_adjustments(final_scores, transcript, audio_profile)
    else:
        logging.warning("Whisper returned an empty transcript; using audio-only result.")
        final_scores = audio_scores

    emotion, confidence = max(final_scores.items(), key=lambda pair: pair[1])
    return {
        "emotion": emotion,
        "confidence": float(confidence),
        "transcript": transcript,
    }
