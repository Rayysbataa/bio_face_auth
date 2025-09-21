# Face Authentication System

A secure, scalable face authentication system using InsightFace (ArcFace) for face recognition. This system is part of a broader multi-modal authentication platform.

## Features

- Face-based user enrollment with multiple angles
- Face verification using ArcFace embeddings
- Anti-spoofing protection
- FAISS-based vector storage
- RESTful API interface
- Web-based demo UI (coming soon)

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for better performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face_auth
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download InsightFace models:
```bash
python -c "import insightface; insightface.model_zoo.get_model('arcface_r100.onnx')"
```

## Configuration

Create a `.env` file in the project root with the following settings:

```env
API_HOST=0.0.0.0
API_PORT=8001
MODEL_PATH=data/models/insightface_model
VECTOR_DB_PATH=data/embeddings
SIMILARITY_THRESHOLD=0.6
```

## Usage

1. Start the API server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

2. API Endpoints:

- Enroll User:
```bash
POST /api/v1/enroll
Content-Type: multipart/form-data
- user_id: string
- images: file[] (1-5 images)
```

- Verify User:
```bash
POST /api/v1/verify
Content-Type: multipart/form-data
- user_id: string
- image: file
```

- Get User Info:
```bash
GET /api/v1/user/{user_id}
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Security Considerations

- The system uses ArcFace for high-accuracy face recognition
- Face embeddings are stored securely using FAISS
- Anti-spoofing measures are implemented
- No raw images are stored, only embeddings

## Future Improvements

- Multi-factor authentication
- Enhanced anti-spoofing
- Real-time processing
- Mobile app support
- Web-based demo UI

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 