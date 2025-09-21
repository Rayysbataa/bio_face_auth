from flask import Flask, render_template, request, jsonify
import requests
import os
from pathlib import Path
import logging
import base64
from io import BytesIO
from PIL import Image
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# API configuration
API_URL = os.environ.get('FACE_API_URL', "http://127.0.0.1:8101")
API_KEY = os.environ.get('API_KEY', "your-secret-key-here")

# Ensure upload directory exists
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def check_api_connection():
    """Check if the Face API is accessible"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        logger.info(f"Face API connection check: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Face API connection check failed: {str(e)}")
        return False

def process_base64_image(base64_string, filename="image.jpg"):
    """Convert base64 image to file-like object"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)

        # Create PIL Image to validate
        img = Image.open(BytesIO(image_data))

        # Convert to RGB if necessary (handles RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to BytesIO
        img_io = BytesIO()
        img.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)

        return img_io, filename
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def index():
    api_status = check_api_connection()
    return render_template('index.html', api_status=api_status, api_url=API_URL)

@app.route('/api/enroll', methods=['POST'])
def enroll():
    try:
        # Check API connection first
        if not check_api_connection():
            return jsonify({
                'error': 'Face authentication API is not accessible. Please ensure it is running.'
            }), 503

        data = request.json
        user_id = data.get('user_id')
        images = data.get('images', [])

        logger.info(f"Enrolling user {user_id} with {len(images)} images")

        if not user_id or not images:
            return jsonify({'error': 'Missing user_id or images'}), 400

        if len(images) < 1:
            return jsonify({'error': 'At least 1 image is required for enrollment'}), 400

        if len(images) > 5:
            return jsonify({'error': 'Maximum 5 images allowed for enrollment'}), 400

        # Process images
        files = []
        for idx, img_data in enumerate(images):
            try:
                img_io, filename = process_base64_image(img_data, f"face_{idx}.jpg")
                files.append(('images', (filename, img_io, 'image/jpeg')))
            except Exception as e:
                logger.error(f"Failed to process image {idx}: {str(e)}")
                return jsonify({'error': f'Invalid image data at index {idx}'}), 400

        # Call the enrollment API
        try:
            response = requests.post(
                f"{API_URL}/api/v1/enroll",
                files=files,
                data={'user_id': user_id},
                timeout=30
            )

            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"API Response: {response.text}")

            # Parse response
            if response.status_code == 200:
                result = response.json()
                return jsonify({
                    'success': True,
                    'message': result.get('message', 'User enrolled successfully')
                }), 200
            else:
                error_detail = response.json().get('detail', 'Enrollment failed')
                return jsonify({
                    'success': False,
                    'error': error_detail
                }), response.status_code

        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return jsonify({'error': 'Request timed out'}), 504
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return jsonify({'error': 'Failed to connect to Face API'}), 503

    except Exception as e:
        logger.error(f"Error during enrollment: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify():
    try:
        # Check API connection first
        if not check_api_connection():
            return jsonify({
                'error': 'Face authentication API is not accessible.'
            }), 503

        data = request.json
        user_id = data.get('user_id')
        image = data.get('image')

        logger.info(f"Verifying user {user_id}")

        if not user_id or not image:
            return jsonify({'error': 'Missing user_id or image'}), 400

        # Process image
        try:
            img_io, filename = process_base64_image(image, "verify.jpg")
            files = [('image', (filename, img_io, 'image/jpeg'))]
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            return jsonify({'error': 'Invalid image data'}), 400

        # Call the verification API
        try:
            response = requests.post(
                f"{API_URL}/api/v1/verify",
                files=files,
                data={'user_id': user_id},
                timeout=30
            )

            logger.info(f"API Response Status: {response.status_code}")
            result = response.json()

            if response.status_code == 200:
                return jsonify({
                    'success': result.get('status') == 'success',
                    'confidence': result.get('confidence', 0),
                    'message': result.get('message', '')
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'confidence': 0,
                    'error': result.get('detail', 'Verification failed')
                }), response.status_code

        except requests.exceptions.Timeout:
            return jsonify({'error': 'Request timed out'}), 504
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return jsonify({'error': 'Failed to connect to Face API'}), 503

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<user_id>', methods=['GET'])
def get_user_info(user_id):
    try:
        response = requests.get(f"{API_URL}/api/v1/user/{user_id}", timeout=10)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        response = requests.delete(f"{API_URL}/api/v1/user/{user_id}", timeout=10)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Failed to delete user: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    app.run(host='0.0.0.0', port=port, debug=True)