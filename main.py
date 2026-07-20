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

def get_working_gemini_models():
    """Query Google API ListModels to discover exact available model identifiers for this API key"""
    if not GEMINI_API_KEY:
        return []
    
    discovered = []
    for ver in ["v1beta", "v1"]:
        try:
            res = requests.get(f"https://generativelanguage.googleapis.com/{ver}/models?key={GEMINI_API_KEY}", timeout=5)
            data = res.json()
            if "models" in data:
                for m in data["models"]:
                    name = m.get("name", "")
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        clean_name = name.replace("models/", "")
                        discovered.append((ver, clean_name))
        except Exception as e:
            print(f"ListModels error for {ver}: {e}")
    
    fallbacks = [
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1beta", "gemini-1.5-flash-001"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.0-pro"),
        ("v1beta", "gemini-pro")
    ]
    for fb in fallbacks:
        if fb not in discovered:
            discovered.append(fb)
            
    return discovered

@app.route('/chat', methods=['POST'])
def process_text_chat():
    """Handle text chat submissions from the Web UI dashboard"""
    global active_model, assistant_name
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
            # Dynamically discover supported model identifiers for this API key
            targets_to_try = get_working_gemini_models()
            
            reply_text = None
            last_error_msg = "No response from Gemini API"
            
            for ver, m_name in targets_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/{ver}/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": (
                                            f"Your name is '{assistant_name}', a friendly ESP32 humanoid robot voice assistant. "
                                            f"Answer the user's question accurately and helpfully. "
                                            f"Keep your response short, conversational, and limited to 1 or 2 sentences. "
                                            f"User asked: {user_text}"
                                        )
                                    }
                                ]
                            }
                        ]
                    }
                    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15)
                    res_json = response.json()
                    
                    if 'candidates' in res_json and res_json['candidates']:
                        candidate = res_json['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            reply_text = candidate['content']['parts'][0]['text'].strip()
                            print(f"Gemini AI ({m_name}) Reply: '{reply_text}'")
                            break
                    
                    if 'error' in res_json:
                        last_error_msg = res_json['error'].get('message', str(res_json['error']))
                        print(f"Gemini API ({m_name}) error: {last_error_msg}")
                    else:
                        print(f"Gemini API ({m_name}) payload without candidates: {res_json}")
                except Exception as inner_e:
                    last_error_msg = str(inner_e)
                    print(f"Gemini model {m_name} exception: {inner_e}")
                    continue

            if not reply_text:
                reply_text = f"I couldn't process that. (Gemini: {last_error_msg})"
        except Exception as e:
            print(f"Gemini API Exception: {e}")
            reply_text = f"API Error: {str(e)}"

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
    global pending_esp_audio
    if pending_esp_audio:
        url = pending_esp_audio
        pending_esp_audio = None  # Reset after sending so it only plays once
        print(f"Delivering pending audio to ESP32: {url}")
        return jsonify({"pending": True, "audio": url})
    return jsonify({"pending": False, "audio": None})

@app.route('/voice', methods=['POST'])
def process_voice():
    """Handle audio upload recordings from the ESP32 hardware client"""
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

    # 2. Query Gemini API directly with dynamically discovered working models
    user_text = "Voice command received"
    reply_text = None
    targets_to_try = get_working_gemini_models()

    for ver, m_name in targets_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/{ver}/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
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
                                    f"Listen to this audio recording. Respond as a friendly humanoid robot voice assistant named '{assistant_name}' (short, 1-2 sentences). "
                                    f"If the user calls you by your name '{assistant_name}' or asks who you are, acknowledge your name politely. "
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
