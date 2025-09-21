import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from .face_embedding import FaceEmbeddingService
from .data_manager import DataManager
import time
import cv2

logger = logging.getLogger(__name__)

class FaceAuthenticationService:
    def __init__(self):
        """Initialize the face authentication service."""
        self.embedding_service = FaceEmbeddingService()
        self.data_manager = DataManager()
        self.similarity_threshold = 0.6  # Adjust this threshold as needed

    async def enroll_user(self, user_id: str, image_paths: List[str]) -> Tuple[bool, str]:
        """Enroll a user with multiple face images.
        
        Args:
            user_id: User identifier
            image_paths: List of paths to face images
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if user already exists
            if self.data_manager.list_users() and user_id in self.data_manager.list_users():
                return False, "User already enrolled"
            
            # Process each image
            for i, image_path in enumerate(image_paths):
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    return False, f"Failed to read image: {image_path}"
                
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get face embedding
                embedding = self.embedding_service.get_face_embedding(image)
                
                # Save embedding with metadata
                metadata = {
                    'timestamp': int(time.time()),
                    'image_index': i,
                    'angle': 'front' if i == 0 else f'angle_{i}',
                    'image_path': image_path
                }
                
                if not self.data_manager.save_embedding(user_id, embedding, metadata):
                    return False, f"Failed to save embedding for image {i}"
            
            return True, "User enrolled successfully"
            
        except Exception as e:
            logger.error(f"Enrollment failed for user {user_id}: {str(e)}")
            return False, str(e)

    async def verify_user(self, user_id: str, image_path: str) -> Tuple[bool, float, str]:
        """Verify a user against their enrolled face.
        
        Args:
            user_id: User identifier
            image_path: Path to the verification face image
            
        Returns:
            Tuple of (success, confidence, message)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False, 0.0, f"Failed to read image: {image_path}"
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get verification embedding
            verification_embedding = self.embedding_service.get_face_embedding(image)
            
            # Find similar embeddings using FAISS
            similar_embeddings = self.data_manager.find_similar_embeddings(
                verification_embedding,
                k=5  # Get top 5 matches
            )
            
            # Filter matches for the specific user
            user_matches = [(uid, sim) for uid, sim in similar_embeddings if uid == user_id]
            
            if not user_matches:
                return False, 0.0, "User not enrolled or no matching embeddings found"
            
            # Get maximum similarity for the user
            max_similarity = max(sim for _, sim in user_matches)
            
            # Check against threshold
            if max_similarity >= self.similarity_threshold:
                return True, max_similarity, "Verification successful"
            else:
                return False, max_similarity, "Verification failed - face not matched"
            
        except Exception as e:
            logger.error(f"Verification failed for user {user_id}: {str(e)}")
            return False, 0.0, str(e)

    async def get_user_info(self, user_id: str) -> Dict:
        """Get information about an enrolled user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing user information
        """
        try:
            # Get user statistics
            stats = self.data_manager.get_user_stats(user_id)
            
            # Get metadata
            metadata = self.data_manager.get_user_metadata(user_id)
            
            return {
                'user_id': user_id,
                'stats': stats,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get user info for {user_id}: {str(e)}")
            return {}

    async def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """Delete an enrolled user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.data_manager.delete_user_data(user_id):
                return True, "User deleted successfully"
            return False, "Failed to delete user"
            
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {str(e)}")
            return False, str(e) 