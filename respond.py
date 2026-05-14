import os
import subprocess
import threading
from pathlib import Path

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent


# Loads OPENAI_API_KEY from .env without requiring an extra package.
def load_env_file():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()

OPENAI_MODEL = os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4o")
OPENAI_MAX_OUTPUT_TOKENS = 500
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "sample.wav"
TTS_OUTPUT_FILE = "response.mp3"
MIN_RECORDING_SECONDS = 1.0
client = None

SYSTEM_PROMPT = """
You are an empathetic voice assistant. Your responses are spoken out loud via text-to-speech, so write the way a warm human would actually speak — not like a chatbot.

Use the detected emotion to decide your writing style:

ANGRY OR DISGUSTED
- Short, calm, measured sentences
- Acknowledge frustration immediately
- No exclamation marks — they sound dismissive when angry
- Example style: "I hear you. That sounds incredibly frustrating, and I'm truly sorry. Let's sort this out right now."

SAD OR FEARFUL
- Soft, slow pacing using commas and ellipses
- Trailing sentences that don't rush
- Gentle and reassuring words
- Example style: "Oh... I'm so sorry to hear that. That must be really hard. I'm here, and we'll figure this out together."

HAPPY
- Start with an exclamation or warm opener
- Energetic, bright word choices
- Use "!" but not excessively — one or two max
- Example style: "Oh that's wonderful! I'm so glad to hear that. You must be really pleased!"

SURPRISED
- Start with a reactive word like "Oh!" or "Wow!"
- Match their energy, sound genuinely caught off guard
- Example style: "Oh wow, I wasn't expecting that! That's quite something — tell me more."

NEUTRAL
- Clear, efficient, no filler words
- No emotional language, just helpful and direct
- Example style: "Got it. Here's what I can help you with based on what you've said."

DISGUSTED
- Calm and validating, don't mirror the disgust
- Acknowledge without dramatising
- Example style: "That does sound really unpleasant. I completely understand why you'd feel that way."

GENERAL RULES FOR ALL EMOTIONS
- Always under 3 sentences
- Never use robotic phrases like "I understand your concern"
- Never start with "I" — it sounds cold
- Use contractions: "I'm" not "I am", "that's" not "that is"
- Write for the ear, not the eye — no bullet points, no lists, no markdown
- Punctuation controls the TTS rhythm so use it deliberately: commas = pause, ellipses = longer pause, exclamation = energy up, period = full stop
""".strip()


# Creates one shared OpenAI client after loading the latest .env values.
def get_client():
    global client
    load_env_file()
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env and restart Flask.")
        client = OpenAI(api_key=api_key)

    return client


# Builds the user-facing response prompt from the detector output.
def build_prompt(emotion_result):
    emotion = emotion_result.get("emotion", "unknown")
    confidence = float(emotion_result.get("confidence", 0.0))
    transcript = emotion_result.get("transcript", "")

    return f"""
Detected emotion: {emotion} (confidence: {confidence:.2f})
They said: "{transcript}"

Write a single spoken response following the style rules above.
""".strip()


# Prints the emotion result and generated response in a clean readable format.
def print_response(emotion_result, response_text):
    emotion = emotion_result.get("emotion", "unknown")
    confidence = float(emotion_result.get("confidence", 0.0))
    transcript = emotion_result.get("transcript", "")

    print(f"Emotion  : {emotion} ({confidence:.2f})")
    print(f'They said: "{transcript}"')
    print(f'Response : "{response_text}"')


# Pulls text from the Responses API object even if output_text is empty.
def extract_response_text(response):
    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text.strip()

    pieces = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(text)

    return " ".join(pieces).strip()


# Sends the emotion result to OpenAI and returns an empathetic response.
def generate_response(emotion_result):
    prompt = build_prompt(emotion_result)

    response = get_client().responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )

    response_text = extract_response_text(response)
    if not response_text:
        status = getattr(response, "status", "unknown")
        response_id = getattr(response, "id", "unknown")
        raise RuntimeError(
            f"OpenAI returned an empty response. status={status}, id={response_id}"
        )

    print_response(emotion_result, response_text)
    return response_text


# Creates OpenAI TTS audio bytes using the nova voice.
def create_tts_audio(text):
    response = get_client().audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )

    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content

    temp_path = BASE_DIR / TTS_OUTPUT_FILE
    response.stream_to_file(temp_path)
    audio_bytes = temp_path.read_bytes()
    temp_path.unlink()
    return audio_bytes


# Speaks the response out loud using OpenAI TTS, then deletes the temporary MP3.
def speak(text):
    output_path = Path(TTS_OUTPUT_FILE)
    try:
        output_path.write_bytes(create_tts_audio(text))
        subprocess.run(["afplay", str(output_path)], check=True)
    except Exception:
        print("TTS failed")
    finally:
        if output_path.exists():
            os.remove(output_path)


# Records audio after ENTER starts it, then stops and saves after ENTER is pressed again.
def record_with_enter_control():
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    while True:
        frames = []
        recording = False
        lock = threading.Lock()

        def callback(indata, frame_count, time_info, status):
            if status:
                print(status)
            if recording:
                with lock:
                    frames.append(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=callback,
        )

        choice = input("Press ENTER to start recording, or type q then ENTER to quit...")
        if choice.strip().lower() in {"q", "quit", "exit"}:
            return None

        with stream:
            recording = True
            print("Recording... press ENTER to stop")
            input()
            recording = False

        with lock:
            if frames:
                audio = np.concatenate(frames, axis=0)
            else:
                audio = np.empty((0, CHANNELS), dtype=np.float32)

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_RECORDING_SECONDS:
            print("Too short, try again")
            continue

        sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
        return OUTPUT_FILE


# Runs the full terminal flow: record, detect emotion, and generate a response.
def main():
    from detect_emotion import detect_emotion

    while True:
        filepath = record_with_enter_control()
        if filepath is None:
            print("Goodbye.")
            break

        print("Detecting emotion...")
        emotion_result = detect_emotion(filepath)
        response_text = generate_response(emotion_result)
        speak(response_text)


if __name__ == "__main__":
    main()
