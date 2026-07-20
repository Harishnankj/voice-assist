import os
import io
import json
import base64
import asyncio
import datetime
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import edge_tts

# Configure Flask template folder and enable bulletproof CORS
root_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=root_dir)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure static directory exists to save response speech files
STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
RESPONSE_AUDIO_PATH = os.path.join(STATIC_DIR, 'response.mp3')

# Server configurations
active_model = "gemini-1.5-flash"  # Default active model
assistant_name = "Jarvis"            # Default assistant call-by-name identity
pending_esp_audio = None             # Audio URL queued for ESP32 hardware playback
chat_history = []                  # In-memory chat transcripts logs
esp_state = "idle"                 # ESP32 hardware state ("idle", "listening", "processing", "speaking")

# Retrieve Gemini API Key from environment (strip spaces, quotes, newlines)
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
if GEMINI_API_KEY:
    masked = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 8 else "too_short"
    print(f"Gemini API configured (Key: {masked}).")
else:
    print("WARNING: GEMINI_API_KEY environment variable is not set.")

@app.route('/api_key_check')
def check_key():
    """Diagnostic route to test Gemini API key validity directly against Google API"""
    if not GEMINI_API_KEY:
        return jsonify({"status": "missing", "message": "GEMINI_API_KEY is not set in Render environment variables"})
    
    masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 8 else "too_short"
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        r = requests.get(url, timeout=5)
        res = r.json()
        if r.status_code == 200 and "models" in res:
            valid_models = [m["name"].replace("models/", "") for m in res["models"] if "generateContent" in m.get("supportedGenerationMethods", [])]
            return jsonify({
                "status": "valid",
                "masked_key": masked_key,
                "available_models": valid_models
            })
        else:
            return jsonify({
                "status": "invalid",
                "masked_key": masked_key,
                "http_status": r.status_code,
                "google_error": res.get("error", {})
            }), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
    return jsonify({"history": chat_history, "esp_state": esp_state})

@app.route('/esp_status', methods=['GET', 'POST'])
def handle_esp_status():
    global esp_state
    if request.method == 'POST':
        data = request.get_json() or {}
        new_state = data.get("state", "").strip()
        if new_state in ["idle", "listening", "processing", "speaking"]:
            esp_state = new_state
            print(f"ESP32 Hardware Status updated: {esp_state}")
        return jsonify({"status": "success", "state": esp_state})
    return jsonify({"state": esp_state})

@app.route('/model', methods=['GET', 'POST'])
def handle_model_select():
    global active_model
    valid_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if request.method == 'POST':
        data = request.get_json() or {}
        model = data.get("model", "").strip()
        if model in valid_models:
            active_model = model
        else:
            active_model = "gemini-1.5-flash"
        print(f"Server active model updated to: {active_model}")
        return jsonify({"status": "success", "active_model": active_model})
    
    return jsonify({"active_model": active_model})

@app.route('/name', methods=['GET', 'POST'])
def handle_name_select():
    global assistant_name
    if request.method == 'POST':
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        if name:
            assistant_name = name
            print(f"Server assistant name updated to: {assistant_name}")
            return jsonify({"status": "success", "assistant_name": assistant_name})
        return jsonify({"error": "Invalid name selection"}), 400
    
    return jsonify({"assistant_name": assistant_name})

WORKING_MODELS_CACHE = []

def get_valid_gemini_models():
    """Query Google API ListModels once to discover exact available model identifiers for this API key"""
    global WORKING_MODELS_CACHE
    if WORKING_MODELS_CACHE:
        return WORKING_MODELS_CACHE

    if not GEMINI_API_KEY:
        return []

    discovered = []
    for ver in ["v1beta", "v1"]:
        try:
            url = f"https://generativelanguage.googleapis.com/{ver}/models?key={GEMINI_API_KEY}"
            r = requests.get(url, timeout=6)
            data = r.json()
            if "models" in data:
                for m in data["models"]:
                    name = m.get("name", "")
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        clean_name = name.replace("models/", "")
                        discovered.append((ver, clean_name))
        except Exception as e:
            print(f"ListModels error for {ver}: {e}")

    defaults = [
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1beta", "gemini-2.0-flash"),
        ("v1beta", "gemini-1.5-pro")
    ]
    for d in defaults:
        if d not in discovered:
            discovered.append(d)

    WORKING_MODELS_CACHE = discovered
    print(f"Discovered working Gemini models for API key: {WORKING_MODELS_CACHE}")
    return WORKING_MODELS_CACHE

def call_gemini_api(prompt_text, inline_audio_b64=None):
    """Call Google Gemini API using dynamically discovered working models for this key"""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY environment variable is not set"

    models_to_try = get_valid_gemini_models()
    last_err = "No response from Gemini API"

    for ver, m_name in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/{ver}/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
            parts = []
            if inline_audio_b64:
                parts.append({
                    "inlineData": {
                        "mimeType": "audio/wav",
                        "data": inline_audio_b64
                    }
                })
            parts.append({"text": prompt_text})

            payload = {"contents": [{"parts": parts}]}
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=12)
            res_json = response.json()

            if 'candidates' in res_json and res_json['candidates']:
                raw_text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"Gemini API Success ({ver}/{m_name}): '{raw_text[:50]}...'")
                return raw_text, None

            if 'error' in res_json:
                last_err = res_json['error'].get('message', str(res_json['error']))
                print(f"Gemini API ({ver}/{m_name}) error: {last_err}")
        except Exception as e:
            last_err = str(e)
            print(f"Gemini API ({ver}/{m_name}) exception: {e}")

    return None, last_err

@app.route('/chat', methods=['POST'])
def process_text_chat():
    """Handle text chat submissions from the Web UI dashboard"""
    global assistant_name
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

    prompt = (
        f"Your name is '{assistant_name}', a friendly ESP32 humanoid robot voice assistant. "
        f"Answer the user's question accurately, directly, and helpfully. "
        f"Keep your response short, conversational, and limited to 1 or 2 sentences. "
        f"User asked: {user_text}"
    )

    raw_text, err = call_gemini_api(prompt)
    if raw_text:
        reply_text = raw_text
    else:
        reply_text = f"I couldn't process that. (Gemini: {err})"

    # Text-to-Speech (TTS) with graceful fallback
    audio_url = None
    try:
        text_to_speech(reply_text, RESPONSE_AUDIO_PATH)
        audio_url = request.url_root + "static/response.mp3"
    except Exception as e:
        print(f"TTS Exception (continuing without audio): {e}")
        audio_url = None
    
    # Queue audio URL so ESP32 hardware plays it out loud on speaker
    global pending_esp_audio
    if audio_url:
        pending_esp_audio = audio_url

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

@app.route('/pending_audio', methods=['GET'])
def get_pending_audio():
    """Endpoint polled by ESP32 hardware to fetch and play web dashboard audio replies"""
    global pending_esp_audio, esp_state
    if pending_esp_audio:
        url = pending_esp_audio
        pending_esp_audio = None  # Reset after sending so it only plays once
        esp_state = "speaking"
        print(f"Delivering pending audio to ESP32: {url}")
        return jsonify({"pending": True, "audio": url})
    return jsonify({"pending": False, "audio": None})

@app.route('/voice', methods=['POST'])
def process_voice():
    """Handle audio upload recordings from the ESP32 hardware client"""
    global assistant_name, esp_state
    esp_state = "processing"
    
    if not request.data:
        return jsonify({"error": "No audio data received"}), 400

    print("Received audio upload from ESP32...")
    audio_data = request.data
    timestamp = get_current_timestamp()
    
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is not configured"}), 500

    # 1. Base64-encode the raw WAV audio
    audio_b64 = base64.b64encode(audio_data).decode('utf-8')

    # 2. Query Gemini API with Call-by-Name Wake Word Filter
    prompt = (
        f"Listen carefully to this audio recording from an ESP32 microphone. "
        f"1. Determine if the user spoke or addressed the robot assistant by its call name '{assistant_name}' (or similar phonetics like Jarvis/Jarves). "
        f"2. Transcribe the exact speech into the 'query' field. "
        f"3. Set 'name_called' to true if the name '{assistant_name}' was called/spoken in the audio, or false if the name was NOT called. "
        f"4. If 'name_called' is true, answer their question in 'reply' as a polite assistant named '{assistant_name}' (1-2 sentences). If false, set 'reply' to null. "
        f"Return your reply ONLY as a valid JSON object containing 'name_called', 'query', and 'reply'."
    )

    raw_text, err = call_gemini_api(prompt, inline_audio_b64=audio_b64)
    name_called = False
    user_text = "Voice command"
    reply_text = None

    if raw_text:
        try:
            clean = raw_text.strip()
            if clean.startswith("```json"): clean = clean[7:]
            if clean.startswith("```"): clean = clean[3:]
            if clean.endswith("```"): clean = clean[:-3]
            clean = clean.strip()

            ai_data = json.loads(clean)
            name_called = bool(ai_data.get("name_called", False))
            user_text = ai_data.get("query", "").strip() or "Voice command"
            reply_text = ai_data.get("reply", "").strip() if name_called else None
            print(f"Gemini Transcribed: '{user_text}' | Name Called ({assistant_name}): {name_called}")
        except Exception as p_err:
            print(f"JSON Parse Error: {p_err}")

    # Wake Word Filter Enforcement: If assistant name was NOT called, ignore request completely!
    if not name_called or not reply_text:
        print(f"[Wake Word Filter] Assistant name '{assistant_name}' was NOT called in audio. Ignoring request.")
        esp_state = "idle"
        return jsonify({
            "status": "ignored",
            "reason": f"Assistant name '{assistant_name}' not called in voice command"
        })

    # Save conversation history logs
    chat_history.append({
        "sender": "user",
        "text": user_text,
        "source": "voice",
        "audio": None,
        "timestamp": timestamp
    })

    # Text-to-Speech (TTS)
    try:
        text_to_speech(reply_text, RESPONSE_AUDIO_PATH)
        print("Speech synthesis completed.")
    except Exception as e:
        print(f"TTS Exception: {e}")
        return jsonify({"error": "Failed to synthesize speech"}), 500

    audio_url = request.url_root + "static/response.mp3"
    global pending_esp_audio
    pending_esp_audio = audio_url

    # Save assistant message to history logs
    chat_history.append({
        "sender": "assistant",
        "text": reply_text,
        "source": "voice",
        "audio": audio_url,
        "timestamp": get_current_timestamp()
    })

    return jsonify({
        "reply": reply_text,
        "audio": audio_url,
        "query": user_text
    })

# Run Flask locally (only for debugging)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
