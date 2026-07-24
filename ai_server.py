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
import time

# Configure Flask template folder and enable bulletproof CORS
root_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=root_dir)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

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
active_provider = "gemini"          # Default AI Provider ("gemini" or "openai")
active_model = "gemini-1.5-flash"  # Default active model
assistant_name = "Persona"           # Default assistant call-by-name identity
pending_esp_audio = None             # Audio URL queued for ESP32 hardware playback
chat_history = []                  # In-memory chat transcripts logs
esp_state = "idle"                 # ESP32 hardware state ("idle", "listening", "processing", "speaking")
esp_last_ping = 0                  # Unix timestamp of last received ESP32 hardware ping

def get_effective_esp_state():
    """Return 'offline' if no ping received from ESP32 for over 10 seconds"""
    global esp_state, esp_last_ping
    if esp_last_ping == 0 or (time.time() - esp_last_ping > 10):
        return "offline"
    return esp_state

# Retrieve Gemini & OpenAI API Keys from environment (strip spaces, quotes, newlines)
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"').strip("'")

if GEMINI_API_KEY:
    masked = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 8 else "configured"
    print(f"Gemini API configured (Key: {masked}).")
else:
    print("WARNING: GEMINI_API_KEY environment variable is not set.")

if OPENAI_API_KEY:
    masked_oai = OPENAI_API_KEY[:4] + "..." + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 8 else "configured"
    print(f"OpenAI API configured (Key: {masked_oai}).")
else:
    print("INFO: OPENAI_API_KEY is not set. OpenAI provider will fall back to Gemini.")

@app.route('/provider', methods=['GET', 'POST'])
def handle_provider_select():
    global active_provider, active_model
    if request.method == 'POST':
        data = request.get_json() or {}
        provider = data.get("provider", "").strip().lower()
        if provider in ["gemini", "openai"]:
            active_provider = provider
            if active_provider == "openai":
                active_model = "gpt-4o-mini"
            else:
                active_model = "gemini-1.5-flash"
            print(f"Server active AI provider updated to: {active_provider} (Model: {active_model})")
            return jsonify({"status": "success", "active_provider": active_provider, "active_model": active_model})
        return jsonify({"error": "Invalid provider"}), 400
    
    return jsonify({"active_provider": active_provider, "active_model": active_model})
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
    """Synthesize high-quality text-to-speech using Microsoft Edge TTS with thread-safe runner"""
    async def _generate():
        communicate = edge_tts.Communicate(text, "en-US-EmmaMultilingualNeural")
        await communicate.save(output_path)

    try:
        asyncio.run(_generate())
    except Exception:
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
    return jsonify({"history": chat_history, "esp_state": get_effective_esp_state()})

@app.route('/esp_status', methods=['GET', 'POST'])
def handle_esp_status():
    global esp_state, esp_last_ping
    if request.method == 'POST':
        esp_last_ping = time.time()
        data = request.get_json() or {}
        new_state = data.get("state", "").strip()
        if new_state in ["idle", "listening", "processing", "speaking"]:
            esp_state = new_state
            print(f"ESP32 Hardware Status updated: {esp_state}")
        return jsonify({"status": "success", "state": get_effective_esp_state()})
    return jsonify({"state": get_effective_esp_state()})

@app.route('/model', methods=['GET', 'POST'])
def handle_model_select():
    global active_model, active_provider
    valid_gemini = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
    valid_openai = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    
    if request.method == 'POST':
        data = request.get_json() or {}
        model = data.get("model", "").strip()
        if model in valid_openai:
            active_provider = "openai"
            active_model = model
        elif model in valid_gemini:
            active_provider = "gemini"
            active_model = model
        print(f"Server active model updated to: {active_model} (Provider: {active_provider})")
        return jsonify({"status": "success", "active_model": active_model, "active_provider": active_provider})
    
    return jsonify({"active_model": active_model, "active_provider": active_provider})

def call_openai_chat_api(prompt_text, system_prompt=None):
    """Call OpenAI Chat Completions API (gpt-4o-mini / gpt-4o)"""
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY is not set in environment"

    if not system_prompt:
        system_prompt = f"You are {assistant_name}, a friendly Alexa-like ESP32 AI voice assistant. Keep your response concise, conversational, and limited to 1 or 2 sentences."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    target_model = active_model if "gpt" in active_model else "gpt-4o-mini"
    
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        res_json = response.json()
        if response.status_code == 200 and 'choices' in res_json and res_json['choices']:
            reply = res_json['choices'][0]['message']['content'].strip()
            print(f"OpenAI API Success ({target_model}): '{reply[:50]}...'")
            return reply, None
        else:
            err_msg = res_json.get('error', {}).get('message', 'OpenAI API Error')
            print(f"OpenAI API Error: {err_msg}")
            return None, err_msg
    except Exception as e:
        print(f"OpenAI API Exception: {e}")
        return None, str(e)

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
        ("v1beta", "gemini-1.5-pro"),
        ("v1beta", "gemini-2.0-flash-exp")
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
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15)  # 15s: handles Gemini variable response times
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
    global assistant_name, active_provider
    data = request.get_json() or {}
    user_text = data.get("text", "").strip()
    
    if not user_text:
        return jsonify({"error": "Message is empty"}), 400
        
    print(f"Received text message from Web: '{user_text}' (Provider: {active_provider})")
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

    if active_provider == "openai" and OPENAI_API_KEY:
        raw_text, err = call_openai_chat_api(user_text, system_prompt=f"You are {assistant_name}, a friendly Alexa-like ESP32 AI voice assistant. Keep your response short and limited to 1-2 sentences.")
    else:
        raw_text, err = call_gemini_api(prompt)

    if not raw_text and active_provider == "openai":
        print(f"OpenAI failed ({err}). Falling back to Gemini...")
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
    global pending_esp_audio, esp_state, esp_last_ping
    esp_last_ping = time.time()
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
    global assistant_name, esp_state, esp_last_ping
    esp_last_ping = time.time()
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

    # Check if request is direct push-to-talk button or bypasses name requirement
    is_direct = request.args.get('direct', '0') in ['1', 'true'] or request.args.get('button', '0') in ['1', 'true']

    # 2. Query Gemini API with dynamic Wake Word / Direct Command Filter
    if is_direct:
        prompt = (
            f"Listen carefully to this audio recording from an ESP32 microphone (Direct command). "
            f"1. Transcribe the exact speech into the 'query' field. "
            f"2. Set 'name_called' to true. "
            f"3. Answer their question or command in 'reply' as a smart Alexa-like voice assistant named '{assistant_name}' (1-2 clear, natural sentences max). "
            f"4. If the audio is empty or pure static/noise, set 'reply' to 'I couldn't hear that clearly. Could you please repeat?'. "
            f"Return your reply ONLY as a valid JSON object containing 'name_called', 'query', and 'reply'."
        )
    else:
        prompt = (
            f"STRICT WAKE WORD FILTER: You are Persona, a smart Alexa-like ESP32 AI voice assistant. "
            f"Listen carefully to the audio clip. "
            f"1. CRITICAL RULE: Check if the speaker explicitly called or addressed the assistant by name '{assistant_name}' (or phonetic variants 'Persona', 'Jarvis'). "
            f"2. If the name '{assistant_name}' or 'Jarvis' was NOT explicitly spoken, OR if the clip is background noise, room chatter, TV sound, or unaddressed statements, you MUST set 'name_called' to false, 'query' to null, and 'reply' to null. DO NOT answer questions unless the assistant name '{assistant_name}' is explicitly called! "
            f"3. ONLY if '{assistant_name}' or 'Jarvis' was explicitly called: set 'name_called' to true, transcribe their question into 'query', and answer their question in 'reply' (1-2 clear, concise sentences max). "
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
            if is_direct:
                name_called = True
            else:
                name_called = bool(ai_data.get("name_called", False))

            user_text = ai_data.get("query", "").strip() if name_called else "Ignored background noise"
            reply_text = ai_data.get("reply", "").strip() if (name_called and ai_data.get("reply")) else None
            print(f"Gemini Transcribed: '{user_text}' | Direct Mode: {is_direct} | Name Called ({assistant_name}): {name_called}")
        except Exception as p_err:
            print(f"JSON Parse Error: {p_err}")

    # Wake Word Filter Enforcement: If assistant name was NOT called and NOT direct mode, ignore request!
    if not name_called or not reply_text:
        print(f"[Wake Word REJECTED] Assistant name '{assistant_name}' was NOT called in audio (Direct: {is_direct}). Ignoring request.")
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
