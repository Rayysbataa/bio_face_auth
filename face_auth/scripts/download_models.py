import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, destination: str):
    """Download a file from URL to destination."""
    try:
        logger.info(f"Downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination)
        logger.info("Download completed successfully")
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise

def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file to the specified directory."""
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Extraction completed successfully")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {str(e)}")
        raise

def setup_models():
    """Download and set up required models."""
    # Create model directory
    model_dir = os.path.join(os.path.expanduser("~"), ".insightface", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Model URLs - using the correct buffalo_l model
    model_urls = {
        "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    }
    
    # Download and extract models
    for model_name, url in model_urls.items():
        try:
            # Download and extract buffalo_l model
            zip_path = os.path.join(model_dir, f"{model_name}.zip")
            download_file(url, zip_path)
            extract_zip(zip_path, model_dir)
            # Clean up zip file
            os.remove(zip_path)
            logger.info(f"Successfully set up {model_name} model")
        except Exception as e:
            logger.error(f"Failed to set up {model_name}: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        setup_models()
        logger.info("All models downloaded and set up successfully")
    except Exception as e:
        logger.error(f"Model setup failed: {str(e)}")
        exit(1) 