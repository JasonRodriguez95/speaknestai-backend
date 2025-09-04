import os
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import re
import base64
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)
# Configurar CORS para permitir solicitudes desde el frontend
CORS(app, resources={r"/*": {"origins": ["https://speaknestai.web.app", "https://speaknestai.firebaseapp.com", "http://localhost:5500"]}})

# ====================================================
# Cargar variables de entorno
# ====================================================
# En local carga .env
# En producción (Render) usará variables de entorno inyectadas
load_dotenv()

# Configurar Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No GEMINI_API_KEY found in environment variables")
genai.configure(api_key=api_key)

# ====================================================
# Funciones auxiliares
# ====================================================
def extract_segments(text):
    """Extrae segmentos marcados con [es] o [en]"""
    pattern = r'\[(es|en)\](.*?)(?=\[|$)'
    segments = re.findall(pattern, text, re.DOTALL)
    return [(lang, text.strip()) for lang, text in segments if text.strip()]

def error_response(message, status_code=400):
    return jsonify({
        'success': False,
        'response': [{'text': f'[color=ff0000]{message}[/color]', 'lang': 'es', 'duration': 0}]
    }), status_code

def audio_to_base64(audio_data):
    """Convierte audio a base64 para Gemini API"""
    try:
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting audio to base64: {str(e)}")
        return None

def process_with_gemini(audio_data, prompt_text):
    try:
        if not audio_data or len(audio_data) == 0:
            return "[color=ff0000]Error: Empty audio data[/color]"

        print(f"Processing audio with Gemini, size: {len(audio_data)} bytes")
        
        audio_base64 = audio_to_base64(audio_data)
        if not audio_base64:
            return "[color=ff0000]Error converting audio to base64[/color]"
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        audio_part = {"mime_type": "audio/webm", "data": audio_base64}
        
        response = model.generate_content([prompt_text, audio_part])
        print(f"Gemini response received: {response.text[:100]}...")
        return response.text
    
    except Exception as e:
        print(f"Error in process_with_gemini: {str(e)}")
        return f"[color=ff0000]Error processing with Gemini: {str(e)}[/color]"

# ====================================================
# Endpoints
# ====================================================
@app.route('/process_intro', methods=['POST', 'OPTIONS'])
def process_intro():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data or 'scenario' not in data:
            return error_response('No scenario provided')
            
        scenario = data.get('scenario', 'friends')
        prompt = f"""
        You are a friendly English conversation partner for Spanish speakers.
        Scenario: {scenario}.
        Generate an introductory message in English with [en] tags and a Spanish translation with [es] tags.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        segments = extract_segments(response.text)
        response_parts = [{'text': text, 'lang': lang, 'duration': 0} for lang, text in segments]
        
        return jsonify({'success': True, 'response': response_parts})
        
    except Exception as e:
        print(f"Server error in process_intro: {str(e)}")
        return error_response(f'Server error: {str(e)}')

@app.route('/process_conversation', methods=['POST', 'OPTIONS'])
def process_conversation():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'audio' not in request.files:
            return error_response('No audio file received')
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return error_response('Empty filename')
            
        scenario = request.form.get('scenario', 'friends')
        last_question = request.form.get('lastQuestion', '')
        
        audio_data = audio_file.read()
        if len(audio_data) < 100:
            return error_response('Audio too short')
        
        prompt = f"""
        You are a friendly English conversation partner for Spanish speakers.
        Scenario: {scenario}.
        Last question: {last_question}.
        Analyze the user's audio input and respond with a conversational reply in English [en] and its Spanish translation [es].
        """
        
        response_text = process_with_gemini(audio_data, prompt)
        segments = extract_segments(response_text)
        response_parts = [{'text': text, 'lang': lang, 'duration': 0} for lang, text in segments]
        
        return jsonify({
            'success': '[color=ff0000]' not in response_text,
            'response': response_parts
        })
        
    except Exception as e:
        print(f"Server error in process_conversation: {str(e)}")
        return error_response(f'Server error: {str(e)}')

@app.route('/process_audio', methods=['POST', 'OPTIONS'])
def process_audio():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'audio' not in request.files:
            return error_response('No audio file received')
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return error_response('Empty filename')
            
        audio_data = audio_file.read()
        if len(audio_data) < 100:
            return error_response('Audio too short')
        
        prompt = """
        Act as a friendly English teacher for Spanish speakers. Analyze the pronunciation and grammar.
        Provide feedback with clear [es] and [en] tags.
        """
        
        feedback_text = process_with_gemini(audio_data, prompt)
        segments = extract_segments(feedback_text)
        response_parts = [{'text': text, 'lang': lang, 'duration': 0} for lang, text in segments]
        
        return jsonify({'success': True, 'feedback': response_parts})
        
    except Exception as e:
        print(f"Server error in process_audio: {str(e)}")
        return error_response(f'Server error: {str(e)}')

@app.route('/verify-admin-key', methods=['POST'])
def verify_admin_key():
    try:
        data = request.get_json()
        if not data or 'adminKey' not in data:
            return jsonify({"success": False, "error": "Clave de administración no proporcionada"}), 400
        admin_key = data['adminKey']
        expected_key = os.getenv("ADMIN_SECRET_KEY")
        if admin_key == expected_key:
            return jsonify({"success": True}), 200
        return jsonify({"success": False, "error": "Clave de administración incorrecta"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('../public', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../public', filename)

# ====================================================
# Main
# ====================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)