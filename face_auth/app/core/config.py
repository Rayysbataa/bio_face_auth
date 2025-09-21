from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    
    # Model Settings
    MODEL_PATH: str = str(Path("data/models/insightface_model"))
    FACE_DETECTION_MODEL: str = "retinaface"  # or "mtcnn"
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./face_auth.db"
    
    # Vector Store Settings
    VECTOR_DB_PATH: str = str(Path("data/embeddings"))
    
    # Face Detection Settings
    MIN_FACE_SIZE: int = 20
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Verification Settings
    SIMILARITY_THRESHOLD: float = 0.6
    MAX_FACES_PER_IMAGE: int = 1
    
    # File Upload Settings
    UPLOAD_DIR: str = str(Path("data/uploads"))
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png"}
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "face_auth.log"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
os.makedirs(settings.MODEL_PATH, exist_ok=True) 