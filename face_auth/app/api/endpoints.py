from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import os
import shutil
from ..services.auth import FaceAuthenticationService
from ..core.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize auth service
auth_service = FaceAuthenticationService()

@router.post("/enroll")
async def enroll_user(
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Enroll a new user with multiple face images.
    
    Args:
        user_id: Unique identifier for the user
        images: List of face images (minimum 1, maximum 5)
        
    Returns:
        Enrollment status and message
    """
    try:
        # Create temporary directory for images
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded images
        image_paths = []
        for image in images:
            # Validate file extension
            ext = image.filename.split('.')[-1].lower()
            if ext not in settings.ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension. Allowed: {settings.ALLOWED_EXTENSIONS}"
                )
            
            # Save file
            file_path = os.path.join(temp_dir, image.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            image_paths.append(file_path)
        
        # Enroll user
        success, message = await auth_service.enroll_user(user_id, image_paths)
        
        # Clean up temporary files
        for path in image_paths:
            os.remove(path)
        os.rmdir(temp_dir)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return {"status": "success", "message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify")
async def verify_user(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Verify a user's face image.
    
    Args:
        user_id: User identifier
        image: Face image for verification
        
    Returns:
        Verification status, confidence score, and message
    """
    try:
        # Save uploaded image
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, image.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Verify user
        success, confidence, message = await auth_service.verify_user(user_id, file_path)
        
        # Clean up temporary file
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        return {
            "status": "success" if success else "failed",
            "confidence": confidence,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}")
async def get_user_info(user_id: str):
    """
    Get user enrollment information.
    
    Args:
        user_id: User identifier
        
    Returns:
        User information if enrolled
    """
    try:
        info = await auth_service.get_user_info(user_id)
        if not info:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"status": "success", "user_info": info}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """Delete an enrolled user."""
    try:
        success, message = await auth_service.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=400, detail=message)
        return {"message": message}
        
    except Exception as e:
        logger.error(f"Failed to delete user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 