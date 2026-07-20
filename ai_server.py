import os
import io
import json
import base64
import asyncio
import datetime
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template
import edge_tts

# Configure Flask template folder to search in the repository root directory
root_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=root_dir)

# Enable CORS manually to allow the webpage hosted on GitHub Pages to communicate with Render/Tunnel
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, bypass-tunnel-reminder")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        return response

@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, bypass-tunnel-reminder")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
    return response

# Ensure static directory exists to save response speech files
STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
RESPONSE_AUDIO_PATH = os.path.join(STATIC_DIR, 'response.mp3')

# Server configurations
active_model = "gemini-1.5-flash"  # Default active model
assistant_name = "Jarvis"            # Default assistant call-by-name identity
pending_esp_audio = None             # Audio URL queued for ESP32 hardware playback
chat_history = []                  # In-memory chat transcripts logs

# Retrieve Gemini API Key from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    print("Gemini API configured successfully.")
else:
    print("WARNING: GEMINI_API_KEY environment variable is not set.")

def text_to_speech(text, output_path):
    """Synthesize high-quality text-to-speech using Microsoft Edge TTS in a dedicated loop"""
    async def _generate():
        communicate = edge_tts.Communicate(text, "en-US-EmmaMultilingualNeural")
        await communicate.save(output_path)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_generate())
    finally:
        loop.close()

def get_current_timestamp():
    """Return HH:MM:SS time string"""
    return datetime.datetime.now().strftime("%H:%M:%S")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": chat_history})

@app.route('/model', methods=['GET', 'POST'])
def handle_model_select():
    global active_model
    if request.method == 'POST':
        data = request.get_json() or {}
        model = data.get("model")
        if model:
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

    if not GEMINI_API_KEY:
        reply_text = "Gemini key is missing. Please configure it in your Render settings."
    else:
        try:
            # Model retry cascade
            models_to_try = [active_model, "gemini-1.5-flash", "gemini-2.0-flash"]
            reply_text = None
            
            for m_name in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": (
                                            f"You are a friendly ESP32 voice assistant. "
                                            f"Keep your response short, conversational, and limited to 1 or 2 sentences. "
                                            f"User asked: {user_text}"
                                        )
                                    }
                                ]
                            }
                        ]
                    }
                    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=10)
                    res_json = response.json()
                    if 'candidates' in res_json and res_json['candidates']:
                        reply_text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                        print(f"Gemini AI ({m_name}) Reply: '{reply_text}'")
                        break
                    else:
                        print(f"Gemini API error payload for {m_name}: {res_json}")
                except Exception as inner_e:
                    print(f"Gemini model {m_name} exception: {inner_e}")
                    continue

            if not reply_text:
                reply_text = "Hello! I am your AI voice assistant. How can I help you today?"
        except Exception as e:
            print(f"Gemini API Exception: {e}")
            reply_text = "Hello! I am your AI voice assistant. How can I help you today?"

    # Text-to-Speech (TTS) with graceful fallback
    audio_url = None
    try:
        text_to_speech(reply_text, RESPONSE_AUDIO_PATH)
        audio_url = request.url_root + "static/response.mp3"
    except Exception as e:
        print(f"TTS Exception (continuing without audio): {e}")
        audio_url = None
    
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

@app.route('/voice', methods=['POST', 'OPTIONS'])
def process_voice():
    """Handle audio upload recordings from the ESP32 hardware client"""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    global active_model
    
    if not request.data:
        return jsonify({"error": "No audio data received"}), 400

    print("Received audio upload from ESP32...")
    audio_data = request.data
    timestamp = get_current_timestamp()
    
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is not configured"}), 500

    # 1. Base64-encode the raw WAV audio
    audio_b64 = base64.b64encode(audio_data).decode('utf-8')

    # 2. Query Gemini API directly with model fallback
    user_text = "Voice command received"
    reply_text = None
    models_to_try = [active_model, "gemini-1.5-flash", "gemini-2.0-flash"]

    for m_name in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/wav",
                                    "data": audio_b64
                                }
                            },
                            {
                                "text": (
                                    "Listen to this audio recording. Respond as a friendly ESP32 voice assistant (short, 1-2 sentences). "
                                    "You must return your reply ONLY as a raw JSON object containing two fields: "
                                    "'query' (the exact text transcription of what the user asked in the audio) and "
                                    "'reply' (your response to their question). Do not include any markdown formatting, backticks, or other text."
                                )
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=20)
            response_json = response.json()
            if 'candidates' in response_json and response_json['candidates']:
                raw_text = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Clean up any potential markdown backticks returned by Gemini
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                
                # Parse the JSON response from Gemini
                ai_data = json.loads(raw_text)
                user_text = ai_data.get("query", "").strip() or "Voice command"
                reply_text = ai_data.get("reply", "").strip()
                
                print(f"Gemini ({m_name}) Transcribed: '{user_text}'")
                print(f"Gemini ({m_name}) Replied: '{reply_text}'")
                break
        except Exception as e:
            print(f"Gemini Voice Exception for {m_name}: {e}")
            continue

    if not reply_text:
        user_text = "Voice command received"
        reply_text = "Hello! I heard your voice command. How can I help you?"

    # 3. Save conversation history logs
    chat_history.append({
        "sender": "user",
        "text": user_text,
        "source": "voice",
        "audio": None,
        "timestamp": timestamp
    })

    # 4. Text-to-Speech (TTS)
    try:
        text_to_speech(reply_text, RESPONSE_AUDIO_PATH)
        print("Speech synthesis completed.")
    except Exception as e:
        print(f"TTS Exception: {e}")
        return jsonify({"error": "Failed to synthesize speech"}), 500

    audio_url = request.url_root + "static/response.mp3"
    
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

# Run Flask locally (only for debugging)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
