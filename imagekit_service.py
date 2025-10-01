import os
import joblib
import tempfile
import requests
import base64
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImageKitService:
    def __init__(self):
        self.private_key = os.getenv('IMAGEKIT_PRIVATE_KEY')
        self.public_key = os.getenv('IMAGEKIT_PUBLIC_KEY')
        self.url_endpoint = os.getenv('IMAGEKIT_URL_ENDPOINT')
        self.models_folder = "ml_models"
        
        if not all([self.private_key, self.public_key, self.url_endpoint]):
            logger.warning("ImageKit credentials not found. Model persistence will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
    
    def save_model(self, model_name: str, model: Any, model_type: str, overwrite: bool = True) -> Optional[str]:
        """Save a trained model to ImageKit and return the file URL"""
        if not self.enabled:
            logger.warning("ImageKit not enabled. Model not saved.")
            return None
            
        temp_file_path = None
        try:
            # Create a temporary file to store the model
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                # Apply compression only for random forest models to reduce file size
                if "random_forest" in model_name:
                    joblib.dump(model, temp_file.name, compress=3)  # moderate zlib compression
                else:
                    joblib.dump(model, temp_file.name)
                temp_file.flush()
                temp_file.close()  # Close the file handle
                temp_file_path = temp_file.name
                
                # Read file content after closing
                with open(temp_file_path, 'rb') as file:
                    file_content = file.read()
                
                # Upload to ImageKit using correct API endpoint
                file_name = f"{model_name}_{model_type}.pkl"
                file_data = base64.b64encode(file_content).decode('utf-8')
                
                # Build form-encoded payload per ImageKit API
                form_data = {
                    "file": file_data,
                    "fileName": file_name,
                    "folder": f"/{self.models_folder}",
                    # string booleans are acceptable in form payloads
                    "useUniqueFileName": "false",
                    "overwriteFile": "true" if overwrite else "false",
                }

                headers = {
                    # Basic auth with private key as username and blank password
                    "Authorization": f"Basic {base64.b64encode(f'{self.private_key}:'.encode('utf-8')).decode('utf-8')}",
                }

                # Use the correct ImageKit API endpoint (form-encoded)
                response = requests.post(
                    "https://api.imagekit.io/v1/files/upload",
                    data=form_data,
                    headers=headers,
                    timeout=30,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    file_url = result.get('url')
                    logger.info(f"Model {model_name} saved to ImageKit: {file_url}")
                    return file_url
                else:
                    logger.error(f"Failed to upload model {model_name}: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return None
        finally:
            # Ensure cleanup happens even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
    
    def load_model(self, model_name: str, model_type: str) -> Optional[Any]:
        """Load a trained model from ImageKit"""
        if not self.enabled:
            logger.warning("ImageKit not enabled. Cannot load model.")
            return None
            
        temp_file_path = None
        try:
            # Search for the model file in ImageKit
            search_params = {
                "searchQuery": f'name="{model_name}_{model_type}.pkl" AND folderPath="/{self.models_folder}"'
            }
            
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.private_key}:'.encode('utf-8')).decode('utf-8')}"
            }
            
            response = requests.get(
                "https://api.imagekit.io/v1/files",
                params=search_params,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(f"Model {model_name}_{model_type} not found in ImageKit")
                return None
            
            result = response.json()
            files = result.get('list', [])
            
            if not files:
                logger.warning(f"Model {model_name}_{model_type} not found in ImageKit")
                return None
            
            # Find the exact model file by name
            target_file = None
            for file_info in files:
                if file_info.get('name') == f"{model_name}_{model_type}.pkl":
                    target_file = file_info
                    break
            
            if not target_file:
                logger.warning(f"Model {model_name}_{model_type} not found in ImageKit")
                return None
            
            # Get the file URL
            file_url = target_file.get('url')
            
            if not file_url:
                logger.error(f"No URL found for model {model_name}_{model_type}")
                return None
            
            # Download and load the model
            download_response = requests.get(file_url)
            download_response.raise_for_status()
            
            # Create temporary file to load the model
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                temp_file.write(download_response.content)
                temp_file.flush()
                temp_file.close()  # Close the file handle
                temp_file_path = temp_file.name
                
                # Load the model after closing
                model = joblib.load(temp_file_path)
                
                logger.info(f"Model {model_name}_{model_type} loaded from ImageKit")
                return model
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}_{model_type}: {str(e)}")
            return None
        finally:
            # Ensure cleanup happens even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
    
    def delete_model(self, model_name: str, model_type: str) -> bool:
        """Delete a model from ImageKit"""
        if not self.enabled:
            logger.warning("ImageKit not enabled. Cannot delete model.")
            return False
            
        try:
            # Search for the model file
            search_params = {
                "searchQuery": f'name="{model_name}_{model_type}.pkl" AND folderPath="/{self.models_folder}"'
            }
            
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.private_key}:'.encode('utf-8')).decode('utf-8')}"
            }
            
            response = requests.get(
                "https://api.imagekit.io/v1/files",
                params=search_params,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(f"Model {model_name}_{model_type} not found for deletion")
                return False
            
            result = response.json()
            files = result.get('list', [])
            
            if not files:
                logger.warning(f"Model {model_name}_{model_type} not found for deletion")
                return False
            
            # Find the exact model file by name
            target_file = None
            for file_info in files:
                if file_info.get('name') == f"{model_name}_{model_type}.pkl":
                    target_file = file_info
                    break
            
            if not target_file:
                logger.warning(f"Model {model_name}_{model_type} not found for deletion")
                return False
            
            # Delete the file
            file_id = target_file.get('fileId')
            
            delete_response = requests.delete(
                f"https://api.imagekit.io/v1/files/{file_id}",
                headers=headers
            )
            
            if delete_response.status_code == 200:
                logger.info(f"Model {model_name}_{model_type} deleted from ImageKit")
                return True
            else:
                logger.error(f"Failed to delete model {model_name}_{model_type}: {delete_response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name}_{model_type}: {str(e)}")
            return False
    
    def list_models(self) -> Dict[str, Any]:
        """List all models in ImageKit"""
        if not self.enabled:
            logger.warning("ImageKit not enabled. Cannot list models.")
            return {}
            
        try:
            search_params = {
                "searchQuery": f'folderPath="/{self.models_folder}"'
            }
            
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.private_key}:'.encode('utf-8')).decode('utf-8')}"
            }
            
            response = requests.get(
                "https://api.imagekit.io/v1/files",
                params=search_params,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                models = {}
                for file_info in result.get('list', []):
                    file_name = file_info.get('name', '')
                    if file_name.endswith('.pkl'):
                        # Extract model name and type from filename
                        name_parts = file_name.replace('.pkl', '').split('_')
                        if len(name_parts) >= 2:
                            model_name = '_'.join(name_parts[:-1])
                            model_type = name_parts[-1]
                            models[f"{model_name}_{model_type}"] = {
                                'url': file_info.get('url'),
                                'created_at': file_info.get('createdAt'),
                                'size': file_info.get('size')
                            }
                return models
            else:
                logger.error(f"Failed to list models: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return {}

# Global ImageKit service instance
imagekit_service = ImageKitService()
