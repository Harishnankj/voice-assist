import os
import io
import asyncio
import nest_asyncio
import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template
import speech_recognition as sr
import google.generativeai as genai
import edge_tts

# Apply nest_asyncio to support running async Edge-TTS inside synchronous Flask
nest_asyncio.apply()

# Configure custom template directory folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Ensure static directory exists to save response speech files
STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
RESPONSE_AUDIO_PATH = os.path.join(STATIC_DIR, 'response.mp3')

# Server configurations
active_model = "gemini-1.5-flash"  # Default active model
chat_history = []                  # In-memory chat transcripts logs

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

def get_current_timestamp():
    """Return HH:MM:SS time string"""
    return datetime.datetime.now().strftime("%H:%M:%S")

@app.route('/')
def index():
    # Serve the beautiful web dashboard
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    # Return conversational log history to the Web client
    return jsonify({"history": chat_history})

@app.route('/model', methods=['GET', 'POST'])
def handle_model_select():
    global active_model
    if request.method == 'POST':
        data = request.get_json() or {}
        model = data.get("model")
        if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
            active_model = model
            print(f"Server active model updated to: {active_model}")
            return jsonify({"status": "success", "active_model": active_model})
        return jsonify({"error": "Invalid model selection"}), 400
    
    return jsonify({"active_model": active_model})

@app.route('/chat', methods=['POST'])
def process_text_chat():
    """Handle text chat submissions from the Web UI dashboard"""
    global active_model
    data = request.get_json() or {}
    user_text = data.get("text", "").strip()
    
    if not user_text:
        return jsonify({"error": "Message is empty"}), 400
        
    print(f"Received text message from Web: '{user_text}'")
    timestamp = get_current_timestamp()
    
    # Save user message to history logs
    chat_history.append({
        "sender": "user",
        "text": user_text,
        "source": "web",
        "audio": None,
        "timestamp": timestamp
    })

    # Query Gemini LLM
    if not GEMINI_API_KEY:
        reply_text = "Gemini key is missing. Please configure it in your Render settings."
    else:
        try:
            model = genai.GenerativeModel(active_model)
            prompt = (
                f"You are a friendly ESP32 voice assistant. "
                f"Keep your response short, conversational, and limited to 1 or 2 sentences. "
                f"User asked: {user_text}"
            )
            response = model.generate_content(prompt)
            reply_text = response.text.strip()
            print(f"Gemini AI ({active_model}) Reply: '{reply_text}'")
        except Exception as e:
            print(f"LLM Exception: {e}")
            reply_text = "Sorry, I had trouble reaching my AI brain. Please try again."

    # Text-to-Speech (TTS)
    try:
        asyncio.run(text_to_speech(reply_text, RESPONSE_AUDIO_PATH))
    except Exception as e:
        print(f"TTS Exception: {e}")
        return jsonify({"error": "Failed to synthesize speech"}), 500

    audio_url = request.url_root + "static/response.mp3"
    
    # Save assistant message to history logs
    chat_history.append({
        "sender": "assistant",
        "text": reply_text,
        "source": "web",
        "audio": audio_url,
        "timestamp": get_current_timestamp()
    })

    return jsonify({
        "reply": reply_text,
        "audio": audio_url
    })

@app.route('/voice', methods=['POST'])
def process_voice():
    """Handle audio upload recordings from the ESP32 hardware client"""
    global active_model
    
    if not request.data:
        return jsonify({"error": "No audio data received"}), 400

    print("Received audio upload from ESP32...")
    audio_data = request.data
    timestamp = get_current_timestamp()
    
    # 1. Speech-to-Text (ASR)
    recognizer = sr.Recognizer()
    try:
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
        # Save user query to history logs
        chat_history.append({
            "sender": "user",
            "text": user_text,
            "source": "voice",
            "audio": None,
            "timestamp": timestamp
        })

        # 2. Query Gemini LLM
        if not GEMINI_API_KEY:
            reply_text = "Gemini key is missing. Please configure it in your Render settings."
        else:
            try:
                model = genai.GenerativeModel(active_model)
                prompt = (
                    f"You are a friendly ESP32 voice assistant. "
                    f"Keep your response short, conversational, and limited to 1 or 2 sentences. "
                    f"User asked: {user_text}"
                )
                response = model.generate_content(prompt)
                reply_text = response.text.strip()
                print(f"Gemini AI ({active_model}) Reply: '{reply_text}'")
            except Exception as e:
                print(f"LLM Exception: {e}")
                reply_text = "Sorry, I had trouble reaching my AI brain. Please try again."

    # 3. Text-to-Speech (TTS)
    try:
        asyncio.run(text_to_speech(reply_text, RESPONSE_AUDIO_PATH))
        print("Speech synthesis completed.")
    except Exception as e:
        print(f"TTS Exception: {e}")
        return jsonify({"error": "Failed to synthesize speech"}), 500

    audio_url = request.url_root + "static/response.mp3"
    
    # Save assistant reply to history logs (only if we successfully parsed a user voice query)
    if user_text:
        chat_history.append({
            "sender": "assistant",
            "text": reply_text,
            "source": "voice",
            "audio": audio_url,
            "timestamp": get_current_timestamp()
        })

    return jsonify({
        "query": user_text,
        "reply": reply_text,
        "audio": audio_url
    })

# Run the Flask app (only for local testing)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
