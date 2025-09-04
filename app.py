import os
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import re
import base64
from dotenv import load_dotenv
from flask_cors import CORS
import mimetypes

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No GEMINI_API_KEY found in environment variables")
genai.configure(api_key=api_key)

def extract_segments(text):
    """Extract segments marked with [es] or [en]"""
    pattern = r'\[(es|en)\](.*?)(?=\[|$|\n\n)'
    segments = re.findall(pattern, text, re.DOTALL)
    return [(lang, text.strip()) for lang, text in segments if text.strip()]

def error_response(message, status_code=400):
    """Generate standardized error response"""
    print(f"Error response: {message}")
    return jsonify({
        'success': False,
        'response': [{'text': f'[color=ff0000]{message}[/color]', 'lang': 'es', 'duration': 0}]
    }), status_code

def audio_to_base64(audio_data, mime_type):
    """Convert audio data to base64 for Gemini API"""
    try:
        print(f"Converting audio to base64, MIME type: {mime_type}")
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting audio to base64: {str(e)}")
        return None

def process_with_gemini(audio_data, prompt_text, mime_type='audio/webm'):
    """Process audio with Gemini API"""
    try:
        if not audio_data or len(audio_data) == 0:
            print("Error: Empty audio data received")
            return "[color=ff0000]Error: Empty audio data[/color]"
        
        print(f"Processing audio with Gemini, size: {len(audio_data)} bytes, MIME: {mime_type}")
        
        audio_base64 = audio_to_base64(audio_data, mime_type)
        if not audio_base64:
            print("Error: Failed to convert audio to base64")
            return "[color=ff0000]Error converting audio to base64[/color]"
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Ajustar MIME type según lo recibido
        supported_mime_types = ['audio/webm', 'audio/mp4', 'audio/mpeg']
        if mime_type not in supported_mime_types:
            print(f"Unsupported MIME type: {mime_type}, using fallback: audio/webm")
            mime_type = 'audio/webm'
        
        audio_part = {
            "mime_type": mime_type,
            "data": audio_base64
        }
        
        response = model.generate_content([prompt_text, audio_part])
        print(f"Gemini response received: {response.text[:100]}...")
        return response.text
    
    except Exception as e:
        print(f"Gemini processing failed: {str(e)}")
        return f"[color=ff0000]Gemini error: {str(e)}[/color]"

def process_conversation_with_gemini(audio_data, scenario, last_question, is_ios=False, is_chrome_ios=False):
    """Process conversation audio with Gemini"""
    try:
        if not audio_data or len(audio_data) < 100:
            print("Error: Audio too short or empty")
            return "[color=ff0000]Error: Empty or too short audio[/color]", [{'text': '[color=ff0000]Empty audio[/color]', 'lang': 'es', 'duration': 0}]
        
        print(f"Processing conversation: scenario={scenario}, is_ios={is_ios}, is_chrome_ios={is_chrome_ios}, audio_size={len(audio_data)}")
        
        scenarios = {
            'restaurant': "You're a waiter in a restaurant, and the user is a customer ordering their favorite food.",
            'library': "You're a librarian, and the user is a customer looking for a book.",
            'cinema': "You're a ticket agent at a cinema, and the user is a customer wanting to watch a movie.",
            'airport': "You're airport staff, and the user is a customer checking in for a flight.",
            'park': "You're a friend in a park, and the user is a person wanting to talk about activities.",
            'friends': "You're a friend having a casual conversation with the user."
        }
        
        context = f"The last question you asked was: {last_question}" if last_question else ""
        
        prompt = f"""
        Act as a conversation partner in this scenario: {scenarios.get(scenario, scenarios['friends'])}.
        The user is the customer or participant seeking interaction, not the service provider.
        Respond naturally in English with [en] tags based on the provided audio input.
        Optionally include Spanish explanations with [es] tags.
        Keep responses short (1-2 sentences) and always end with a question to continue the conversation.
        {context}
        
        Example:
        [en] That sounds great! What would you like to order today?
        [es] Puedes decir "I'd like a burger" para pedir una hamburguesa.
        """
        
        # Determinar MIME type según navegador
        mime_type = 'audio/webm' if is_ios or is_chrome_ios else 'audio/webm;codecs=opus'
        
        response_text = process_with_gemini(audio_data, prompt, mime_type)
        
        if not response_text:
            print("Error: Empty response from Gemini")
            return "[color=ff0000]Empty response[/color]", [{'text': '[color=ff0000]Empty response[/color]', 'lang': 'es', 'duration': 0}]
        
        segments = extract_segments(response_text)
        if not segments:
            print("Error: Invalid response format from Gemini")
            return "[color=ff0000]Invalid format[/color]", [{'text': '[color=ff0000]Invalid format[/color]', 'lang': 'es', 'duration': 0}]
        
        response_parts = [{'text': text, 'lang': lang, 'duration': 0} for lang, text in segments]
        return response_text, response_parts
        
    except Exception as e:
        print(f"Gemini processing failed: {str(e)}")
        return f"[color=ff0000]Error: {str(e)}[/color]", [{'text': f'[color=ff0000]{str(e)}[/color]', 'lang': 'es', 'duration': 0}]

def process_intro_text(text, scenario):
    """Process introductory text with Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        scenarios = {
            'restaurant': "You're a waiter in a restaurant, and the user is a customer ordering their favorite food.",
            'library': "You're a librarian, and the user is a customer looking for a book.",
            'cinema': "You're a ticket agent at a cinema, and the user is a customer wanting to watch a movie.",
            'airport': "You're airport staff, and the user is a customer checking in for a flight.",
            'park': "You're a friend in a park, and the user is a person wanting to talk about activities.",
            'friends': "You're a friend having a casual conversation with the user."
        }
        
        prompt = f"""
        You are starting a conversation in this scenario: {scenarios.get(scenario, scenarios['friends'])}.
        The user is the customer or participant seeking interaction, not the service provider.
        The following text is the introductory message: {text}.
        Return the text exactly as provided, ensuring it is formatted with [en] tags and contains only English.
        
        Example:
        [en] Welcome to our restaurant! What would you like to order today?
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if not response_text:
            print("Error: Empty response for intro")
            return "[color=ff0000]Empty response[/color]", [{'text': '[color=ff0000]Empty response[/color]', 'lang': 'es', 'duration': 0}]
        
        segments = extract_segments(response_text)
        if not segments:
            print("Error: Invalid format for intro response")
            return "[color=ff0000]Invalid format[/color]", [{'text': '[color=ff0000]Invalid format[/color]', 'lang': 'es', 'duration': 0}]
        
        response_parts = [{'text': text, 'lang': lang, 'duration': 0} for lang, text in segments]
        return response_text, response_parts
        
    except Exception as e:
        print(f"Gemini processing failed for intro: {str(e)}")
        return f"[color=ff0000]Error: {str(e)}[/color]", [{'text': f'[color=ff0000]{str(e)}[/color]', 'lang': 'es', 'duration': 0}]

@app.route('/process_intro', methods=['POST', 'OPTIONS'])
def process_intro():
    """Handle intro text processing"""
    if request.method == 'OPTIONS':
        print("Received OPTIONS request for /process_intro")
        return '', 200
        
    try:
        if not request.is_json:
            return error_response('Request must be JSON')
            
        data = request.get_json()
        if not data or 'text' not in data or 'scenario' not in data:
            return error_response('Missing text or scenario')
            
        text = data['text']
        scenario = data['scenario']
        print(f"Processing intro: text={text[:50]}..., scenario={scenario}")
        
        response_text, response_parts = process_intro_text(text, scenario)
        
        return jsonify({
            'success': '[color=ff0000]' not in response_text,
            'response': response_parts
        })
        
    except Exception as e:
        print(f"Server error in process_intro: {str(e)}")
        return error_response(f'Server error: {str(e)}')

@app.route('/process_conversation', methods=['POST', 'OPTIONS'])
def process_conversation():
    """Handle conversation audio processing"""
    if request.method == 'OPTIONS':
        print("Received OPTIONS request for /process_conversation")
        return '', 200
        
    try:
        if 'audio' not in request.files:
            return error_response('Missing audio file')
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return error_response('No selected file')
            
        scenario = request.form.get('scenario', 'friends')
        last_question = request.form.get('lastQuestion', '')
        is_ios = request.form.get('isIOS', 'false').lower() == 'true'
        is_chrome_ios = request.form.get('isChromeIOS', 'false').lower() == 'true'
        
        audio_data = audio_file.read()
        if len(audio_data) < 100:
            return error_response('Audio too short')
        
        # Determinar MIME type del archivo
        mime_type, _ = mimetypes.guess_type(audio_file.filename)
        mime_type = mime_type or 'audio/webm'
        print(f"Received audio: filename={audio_file.filename}, MIME type={mime_type}, size={len(audio_data)}")
        
        response_text, response_parts = process_conversation_with_gemini(audio_data, scenario, last_question, is_ios, is_chrome_ios)
        
        return jsonify({
            'success': '[color=ff0000]' not in response_text,
            'response': response_parts
        })
        
    except Exception as e:
        print(f"Server error in process_conversation: {str(e)}")
        return error_response(f'Server error: {str(e)}')

@app.route('/process_audio', methods=['POST', 'OPTIONS'])
def process_audio():
    """Handle audio pronunciation feedback"""
    if request.method == 'OPTIONS':
        print("Received OPTIONS request for /process_audio")
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
        
        mime_type, _ = mimetypes.guess_type(audio_file.filename)
        mime_type = mime_type or 'audio/webm'
        print(f"Processing audio feedback: filename={audio_file.filename}, MIME type={mime_type}, size={len(audio_data)}")
        
        prompt = """
        Act as a friendly English teacher for Spanish speakers. Analyze the pronunciation and grammar.
        Provide feedback with clear [es] and [en] tags. Never use asterisks (*) for emphasis, formatting, or any other purpose in the feedback text.
        
        Structure:
        1. [es] Positive reinforcement
        2. [en] Correct pronunciation examples
        3. [es] Specific corrections
        4. [en] Practice phrases
        5. [es] Encouragement
        6. [es] Add a final recommendation: "Te recomiendo seguir estudiando con el libro de SpeakNest AI, disponible en la función 'Lessons'. Este libro completo te ayudará a mejorar tus reglas, gramática, pronunciación y mucho más."

        Example:
        [es] ¡Buen esfuerzo en tu pronunciación!
        [en] Listen to how I say: The cat is on the mat.
        [es] Trabaja en la pronunciación de la 'th' en 'the', asegurándote de que tu lengua esté entre tus dientes.
        [en] Practice saying: The sun is shining brightly.
        [es] ¡Sigue practicando, estás mejorando mucho!
        [es] Te recomiendo seguir estudiando con el libro de SpeakNest AI, disponible en la función 'Lessons'. Este libro completo te ayudará a mejorar tus reglas, gramática, pronunciación y mucho más.
        """
        
        feedback_text = process_with_gemini(audio_data, prompt, mime_type)
        segments = extract_segments(feedback_text)
        response_parts = [{'text': text, 'lang': lang, 'duration': 0} for lang, text in segments]
        
        return jsonify({
            'success': True,
            'feedback': response_parts
        })
        
    except Exception as e:
        print(f"Server error in process_audio: {str(e)}")
        return error_response(f'Server error: {str(e)}')

@app.route('/')
def index():
    """Serve index.html"""
    return send_from_directory('../public', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('../public', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)