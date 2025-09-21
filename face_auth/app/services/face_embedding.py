import insightface
import cv2
import numpy as np
import os
import logging
from pathlib import Path
import urllib.request
import zipfile
import shutil
from typing import Tuple, List, Optional, Dict
from ..core.config import settings

logger = logging.getLogger(__name__)

class FaceEmbeddingService:
    def __init__(self):
        """Initialize the face embedding service."""
        self.model_path = os.path.expanduser("~/.insightface/models")
        
        try:
            # Initialize detector with CPU provider
            self.detector = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root=self.model_path,
                providers=['CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=-1, det_size=(640, 640))
            
            # Initialize recognizer
            self.recognizer = insightface.model_zoo.get_model(
                'buffalo_l',
                root=self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            logger.info("Face embedding service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face embedding service: {str(e)}")
            raise RuntimeError(f"Failed to initialize face embedding service: {str(e)}")

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            faces = self.detector.get(image)
            if not faces:
                raise ValueError("No face detected in the image")
            return faces
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise

    def get_face_embedding(self, image: np.ndarray) -> np.ndarray:
        """Get face embedding from an image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Face embedding vector
        """
        try:
            # Detect faces
            faces = self.detector.get(image)
            if not faces:
                raise ValueError("No face detected in the image")
            
            # Get the first face
            face = faces[0]
            
            # Get embedding
            embedding = self.recognizer.get(image, face)
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get face embedding: {str(e)}")
            raise

    def get_face_embeddings(self, image: np.ndarray) -> list:
        """Get face embeddings for all faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face embedding vectors
        """
        try:
            # Detect faces
            faces = self.detect_faces(image)
            if not faces:
                raise ValueError("No face detected in the image")
            
            # Get embeddings for all faces
            embeddings = []
            for face in faces:
                embedding = self.recognizer.get(image, face)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get face embeddings: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for face detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def align_face(self, image: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        """
        Align face using facial landmarks.
        
        Args:
            image: Input image
            face: Detected face information
            
        Returns:
            Aligned face image or None if alignment fails
        """
        try:
            # Get facial landmarks
            landmarks = face.kps
            
            # Calculate alignment matrix
            src = landmarks.astype(np.float32)
            dst = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            
            # Get alignment matrix
            M = cv2.estimateAffinePartial2D(src, dst)[0]
            
            # Align face
            aligned_face = cv2.warpAffine(
                image, M, (112, 112), borderValue=0.0
            )
            
            return aligned_face
        except Exception as e:
            logger.error(f"Face alignment failed: {str(e)}")
            return None
    
    def generate_embedding(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Generate face embedding from an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (embedding, error_message)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to read image"
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detect_faces(image)
            if not faces:
                return None, "No face detected in image"
            
            # Get the face with highest confidence
            face = faces[0]
            
            # Align face
            aligned_face = self.align_face(image, face)
            if aligned_face is None:
                return None, "Face alignment failed"
            
            # Generate embedding
            embedding = self.recognizer.get_embedding(aligned_face)
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding, None
            
        except Exception as e:
            logger.error(f"Failed to generate face embedding: {str(e)}")
            return None, str(e)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            raise 