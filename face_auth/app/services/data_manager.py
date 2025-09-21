import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import faiss
import pickle

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, base_path: str = None):
        """Initialize the data manager.
        
        Args:
            base_path: Base path for data storage. If None, uses default path.
        """
        self.base_path = base_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.models_path = os.path.join(self.base_path, "models")
        self.embeddings_path = os.path.join(self.base_path, "embeddings")
        self.index_path = os.path.join(self.base_path, "faiss_index")
        
        # Create necessary directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 512  # ArcFace embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.user_map = {}  # Maps FAISS indices to user IDs
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"Data manager initialized with base path: {self.base_path}")

    def _load_index(self):
        """Load existing FAISS index and user mapping."""
        try:
            index_file = os.path.join(self.index_path, "face_index.faiss")
            mapping_file = os.path.join(self.index_path, "user_mapping.pkl")
            
            if os.path.exists(index_file) and os.path.exists(mapping_file):
                self.index = faiss.read_index(index_file)
                with open(mapping_file, 'rb') as f:
                    self.user_map = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")

    def _save_index(self):
        """Save FAISS index and user mapping."""
        try:
            index_file = os.path.join(self.index_path, "face_index.faiss")
            mapping_file = os.path.join(self.index_path, "user_mapping.pkl")
            
            faiss.write_index(self.index, index_file)
            with open(mapping_file, 'wb') as f:
                pickle.dump(self.user_map, f)
            logger.info("Saved FAISS index and user mapping")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")

    def save_embedding(self, user_id: str, embedding: np.ndarray, metadata: Dict = None) -> bool:
        """Save a face embedding for a user.
        
        Args:
            user_id: User identifier
            embedding: Face embedding vector
            metadata: Additional metadata (e.g., capture time, angle)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create user directory if it doesn't exist
            user_dir = os.path.join(self.embeddings_path, user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = metadata.get('timestamp', '') if metadata else ''
            filename = f"embedding_{timestamp}.npy"
            filepath = os.path.join(user_dir, filename)
            
            # Save embedding to file
            np.save(filepath, embedding)
            
            # Save metadata
            if metadata:
                metadata_path = os.path.join(user_dir, f"metadata_{timestamp}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
            
            # Add to FAISS index
            embedding = embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(embedding)  # Normalize for cosine similarity
            self.index.add(embedding)
            
            # Update user mapping
            self.user_map[self.index.ntotal - 1] = user_id
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Saved embedding for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embedding for user {user_id}: {str(e)}")
            return False

    def find_similar_embeddings(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find similar embeddings using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar embeddings to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        try:
            # Prepare query
            query = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query)
            
            # Search
            similarities, indices = self.index.search(query, k)
            
            # Map indices to user IDs
            results = []
            for idx, sim in zip(indices[0], similarities[0]):
                if idx != -1:  # Valid index
                    user_id = self.user_map.get(idx)
                    if user_id:
                        results.append((user_id, float(sim)))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {str(e)}")
            return []

    def get_embeddings(self, user_id: str) -> List[np.ndarray]:
        """Get all embeddings for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of face embeddings
        """
        try:
            user_dir = os.path.join(self.embeddings_path, user_id)
            if not os.path.exists(user_dir):
                return []
            
            embeddings = []
            for file in os.listdir(user_dir):
                if file.endswith('.npy'):
                    embedding = np.load(os.path.join(user_dir, file))
                    embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings for user {user_id}: {str(e)}")
            return []

    def get_user_metadata(self, user_id: str) -> List[Dict]:
        """Get metadata for all embeddings of a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of metadata dictionaries
        """
        try:
            user_dir = os.path.join(self.embeddings_path, user_id)
            if not os.path.exists(user_dir):
                return []
            
            metadata_list = []
            for file in os.listdir(user_dir):
                if file.endswith('.json'):
                    with open(os.path.join(user_dir, file), 'r') as f:
                        metadata = json.load(f)
                        metadata_list.append(metadata)
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Failed to get metadata for user {user_id}: {str(e)}")
            return []

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete file-based data
            user_dir = os.path.join(self.embeddings_path, user_id)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
            
            # Remove from FAISS index
            indices_to_remove = [idx for idx, uid in self.user_map.items() if uid == user_id]
            if indices_to_remove:
                # Create new index without removed vectors
                new_index = faiss.IndexFlatIP(self.dimension)
                new_user_map = {}
                
                # Rebuild index excluding removed vectors
                for idx in range(self.index.ntotal):
                    if idx not in indices_to_remove:
                        vector = self.index.reconstruct(idx)
                        new_index.add(vector.reshape(1, -1))
                        new_user_map[new_index.ntotal - 1] = self.user_map[idx]
                
                self.index = new_index
                self.user_map = new_user_map
                self._save_index()
            
            logger.info(f"Deleted data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data for user {user_id}: {str(e)}")
            return False

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the path to a model file.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Path to the model file or None if not found
        """
        model_path = os.path.join(self.models_path, model_name)
        return model_path if os.path.exists(model_path) else None

    def list_users(self) -> List[str]:
        """List all enrolled users.
        
        Returns:
            List of user IDs
        """
        try:
            return list(set(self.user_map.values()))
        except Exception as e:
            logger.error(f"Failed to list users: {str(e)}")
            return []

    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user's embeddings.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing user statistics
        """
        try:
            user_dir = os.path.join(self.embeddings_path, user_id)
            if not os.path.exists(user_dir):
                return {}
            
            embedding_files = [f for f in os.listdir(user_dir) if f.endswith('.npy')]
            metadata_files = [f for f in os.listdir(user_dir) if f.endswith('.json')]
            
            # Count embeddings in FAISS index
            faiss_count = sum(1 for uid in self.user_map.values() if uid == user_id)
            
            return {
                'user_id': user_id,
                'embedding_count': len(embedding_files),
                'metadata_count': len(metadata_files),
                'faiss_index_count': faiss_count,
                'last_updated': max(os.path.getmtime(os.path.join(user_dir, f)) 
                                  for f in embedding_files) if embedding_files else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for user {user_id}: {str(e)}")
            return {} 