import os
import io
import asyncio
import nest_asyncio
from flask import Flask, request, jsonify, send_from_directory
import speech_recognition as sr
import google.generativeai as genai
import edge_tts

# Apply nest_asyncio to support running async Edge-TTS inside synchronous Flask
nest_asyncio.apply()

app = Flask(__name__)

# Ensure static directory exists to save response speech files
STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
RESPONSE_AUDIO_PATH = os.path.join(STATIC_DIR, 'response.mp3')

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("WARNING: GEMINI_API_KEY environment variable is not set.")

async def text_to_speech(text, output_path):
    """Synthesize high-quality text-to-speech using Microsoft Edge TTS"""
    # Using 'en-US-EmmaMultilingualNeural' (A friendly, natural female voice)
    communicate = edge_tts.Communicate(text, "en-US-EmmaMultilingualNeural")
    await communicate.save(output_path)

@app.route('/')
def index():
    return "Xiaozhi AI Custom Voice Assistant Cloud Server is running!"

@app.route('/voice', methods=['POST'])
def process_voice():
    # Verify that audio file is present in request
    if not request.data:
        return jsonify({"error": "No audio data received"}), 400

    print("Received audio upload from ESP32...")
    audio_data = request.data
    
    # 1. Speech-to-Text (ASR)
    recognizer = sr.Recognizer()
    try:
        # Load the uploaded audio bytes into a speech recognition AudioFile
        audio_file = io.BytesIO(audio_data)
        with sr.AudioFile(audio_file) as source:
            audio_recording = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_recording)
            print(f"Transcribed Text: '{user_text}'")
    except sr.UnknownValueError:
        print("ASR Error: Google Speech Recognition could not understand audio")
        user_text = ""
    except Exception as e:
        print(f"ASR Exception: {e}")
        user_text = ""

    if not user_text:
        reply_text = "I couldn't hear you clearly. Could you please repeat that?"
    else:
        # 2. Query Gemini LLM
        if not GEMINI_API_KEY:
            reply_text = "Gemini key is missing. Please set the GEMINI API KEY on your cloud dashboard."
        else:
            try:
                # Use Gemini 1.5 Flash (fast and lightweight)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"You are a friendly ESP32 voice assistant. "
                    f"Keep your response short, conversational, and limited to 1 or 2 sentences. "
                    f"User asked: {user_text}"
                )
                response = model.generate_content(prompt)
                reply_text = response.text.strip()
                print(f"Gemini AI Reply: '{reply_text}'")
            except Exception as e:
                print(f"LLM Exception: {e}")
                reply_text = "Sorry, I had trouble reaching my AI brain. Please try again."

    # 3. Text-to-Speech (TTS)
    try:
        # Run async speech generator in Flask's synchronous loop
        asyncio.run(text_to_speech(reply_text, RESPONSE_AUDIO_PATH))
        print("Speech synthesis completed.")
    except Exception as e:
        print(f"TTS Exception: {e}")
        # Return fallback error
        return jsonify({"error": "Failed to synthesize speech"}), 500

    # 4. Generate the audio stream URL
    # request.url_root will automatically create the correct public domain HTTP/HTTPS root
    audio_url = request.url_root + "static/response.mp3"
    print(f"Returning audio response URL: {audio_url}")

    return jsonify({
        "query": user_text,
        "reply": reply_text,
        "audio": audio_url
    })

# Run the Flask app (only for local testing; Render will use gunicorn)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
