from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/v1/enroll', methods=['POST'])
def enroll():
    try:
        # Forward the request to the FastAPI backend
        files = request.files.getlist('images')
        user_id = request.form.get('user_id')
        
        # Prepare multipart form data
        form_data = {'user_id': user_id}
        files_data = [('images', (f'image_{i}.jpg', file)) for i, file in enumerate(files)]
        
        # Send request to FastAPI backend
        response = requests.post(
            f'{API_BASE_URL}/api/v1/enroll',
            data=form_data,
            files=files_data
        )
        
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/v1/verify', methods=['POST'])
def verify():
    try:
        # Forward the request to the FastAPI backend
        image = request.files.get('image')
        user_id = request.form.get('user_id')
        
        # Prepare multipart form data
        form_data = {'user_id': user_id}
        files_data = [('image', ('verify.jpg', image))]
        
        # Send request to FastAPI backend
        response = requests.post(
            f'{API_BASE_URL}/api/v1/verify',
            data=form_data,
            files=files_data
        )
        
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 