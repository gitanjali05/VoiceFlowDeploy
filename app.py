import os
from pathlib import Path

from dotenv import load_dotenv
from io import BytesIO

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMP_AUDIO_FILE = BASE_DIR / "temp_audio.wav"
DEFAULT_RESPONSE = "I'm sorry, I couldn't process that. Please try again."

app = Flask(__name__)
CORS(app)


# Builds a local fallback reply if the OpenAI response call fails.
def fallback_response(emotion_result):
    emotion = emotion_result.get("emotion", "neutral")
    transcript = emotion_result.get("transcript", "")

    if emotion in {"angry", "disgusted"}:
        return "I'm sorry this has been so frustrating. I understand why you're upset, and I'm here to help fix it."
    if emotion in {"sad", "fearful"}:
        return "I'm really sorry you're feeling this way. Let's take this step by step, and I'll help you through it."
    if emotion == "surprised":
        return "I can understand why that caught you off guard. Let's look at what happened and sort it out."
    if transcript:
        return "Thanks for explaining that. I'm here to help you work through it."
    return DEFAULT_RESPONSE


# Silences the browser's automatic favicon request.
@app.get("/favicon.ico")
def favicon():
    return "", 204


# Serves the frontend page so the browser can load the voice detector app.
@app.get("/")
def index():
    index_file = BASE_DIR / "index.html"
    fallback_file = BASE_DIR / "voiceflow.html"
    frontend_file = index_file if index_file.exists() else fallback_file
    return send_file(frontend_file)


# Receives browser audio, runs emotion detection, generates an AI response, and returns JSON.
@app.post("/analyze")
def analyze():
    try:
        if "audio" not in request.files:
            raise ValueError("Missing audio file in FormData key 'audio'.")

        audio_file = request.files["audio"]
        audio_file.save(TEMP_AUDIO_FILE)

        from detect_emotion import detect_emotion
        from respond import generate_response

        emotion_result = detect_emotion(str(TEMP_AUDIO_FILE))
        try:
            response_text = generate_response(emotion_result)
        except Exception as error:
            print(f"Response generation failed: {error}")
            response_text = fallback_response(emotion_result)

        return jsonify(
            {
                "emotion": emotion_result.get("emotion", "neutral"),
                "confidence": float(emotion_result.get("confidence", 0.0)),
                "response": response_text,
            }
        )
    except Exception as error:
        print(f"Analyze failed: {error}")
        return (
            jsonify(
                {
                    "error": "something went wrong",
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "response": DEFAULT_RESPONSE,
                }
            ),
            500,
        )
    finally:
        if TEMP_AUDIO_FILE.exists():
            TEMP_AUDIO_FILE.unlink()


# Creates OpenAI TTS audio for the browser to play.
@app.post("/tts")
def tts():
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            raise ValueError("Missing text for TTS.")

        from respond import create_tts_audio

        audio_bytes = create_tts_audio(text)
        return send_file(
            BytesIO(audio_bytes),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="response.mp3",
        )
    except Exception as error:
        print(f"TTS failed: {error}")
        return jsonify({"error": "tts failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
