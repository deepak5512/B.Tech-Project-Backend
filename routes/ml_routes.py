from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
import logging

from ml_service import ml_service
from ml_models import ModelEvaluation, PredictionResult, TrainingJob
import crud

router = APIRouter(prefix="/ml", tags=["Machine Learning"])
logger = logging.getLogger(__name__)

# In-memory storage for training jobs (in production, use Redis or database)
training_jobs = {}

@router.post("/train")
async def train_models(data: List[Dict[str, Any]]):
    """Train all ML models on the provided data"""
    try:
        logger.info(f"Training models on {len(data)} records")
        
        # Clear existing models before training new ones
        logger.info("Clearing existing models...")
        ml_service.clear_models()
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Train models
        results = ml_service.train_models(df)
        
        # Save evaluation results to database
        evaluation_records = []
        
        # Save classifier results
        for model_name, metrics in results['classifiers'].items():
            if 'error' not in metrics:
                evaluation = ModelEvaluation(
                    model_name=model_name,
                    model_type='classification',
                    accuracy=metrics.get('accuracy'),
                    precision=metrics.get('precision'),
                    recall=metrics.get('recall'),
                    f1_score=metrics.get('f1_score'),
                    training_time=metrics.get('training_time', 0)
                )
                evaluation_records.append(evaluation)
        
        # Save regressor results
        for model_name, metrics in results['regressors'].items():
            if 'error' not in metrics:
                evaluation = ModelEvaluation(
                    model_name=model_name,
                    model_type='regression',
                    mse=metrics.get('mse'),
                    mae=metrics.get('mae'),
                    r2_score=metrics.get('r2_score'),
                    training_time=metrics.get('training_time', 0)
                )
                evaluation_records.append(evaluation)
        
        # Save to database (you'll need to implement this in crud.py)
        for evaluation in evaluation_records:
            crud.save_model_evaluation(evaluation)
        
        return {
            "message": "Models trained successfully",
            "results": results,
            "evaluations_saved": len(evaluation_records)
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/{model_name}")
async def predict(model_name: str, data: Dict[str, Any]):
    """Make prediction using a specific model with JSON data"""
    try:
        logger.info(f"Making prediction with {model_name}")
        
        # Make prediction
        result = ml_service.predict(data, model_name)
        
        # Save prediction result to database
        prediction = PredictionResult(
            model_name=model_name,
            model_type=result['model_type'],
            predicted_type=result.get('predicted_type'),
            predicted_concentration=result.get('predicted_concentration'),
            confidence=result.get('confidence'),
            input_data=data
        )
        
        crud.save_prediction_result(prediction)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-file/{model_name}")
async def predict_file(model_name: str, file: UploadFile = File(...)):
    """Make prediction using a specific model with file upload"""
    try:
        logger.info(f"Making prediction with {model_name} from file")
        
        # Read and parse file
        content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'json':
            data = json.loads(content.decode('utf-8'))
            # Handle both single object and array
            if isinstance(data, list):
                data = data[0]  # Take first item for single prediction
        elif file_extension == 'csv':
            import pandas as pd
            import io
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            # Convert first row to dict
            data = df.iloc[0].to_dict()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use JSON or CSV.")
        
        # Make prediction
        result = ml_service.predict(data, model_name)
        
        # Save prediction result to database
        prediction = PredictionResult(
            model_name=model_name,
            model_type=result['model_type'],
            predicted_type=result.get('predicted_type'),
            predicted_concentration=result.get('predicted_concentration'),
            confidence=result.get('confidence'),
            input_data=data
        )
        
        crud.save_prediction_result(prediction)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch/{model_name}")
async def predict_batch(model_name: str, data: List[Dict[str, Any]]):
    """Make predictions on multiple data points"""
    try:
        logger.info(f"Making batch predictions with {model_name} on {len(data)} records")
        
        results = []
        for item in data:
            try:
                result = ml_service.predict(item, model_name)
                results.append({
                    "input": item,
                    "prediction": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": item,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "model_name": model_name,
            "total_predictions": len(data),
            "successful_predictions": len([r for r in results if r['success']]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models():
    """Get information about trained models"""
    try:
        model_info = ml_service.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations")
async def get_evaluations():
    """Get all model evaluation results"""
    try:
        evaluations = crud.get_all_evaluations()
        return {"evaluations": evaluations}
    except Exception as e:
        logger.error(f"Error getting evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations/{model_name}")
async def get_model_evaluations(model_name: str):
    """Get evaluation results for a specific model"""
    try:
        evaluations = crud.get_model_evaluations(model_name)
        return {"evaluations": evaluations}
    except Exception as e:
        logger.error(f"Error getting evaluations for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions")
async def get_predictions():
    """Get all prediction results"""
    try:
        predictions = crud.get_all_predictions()
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error getting predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/load")
async def load_models():
    """Load all models from ImageKit storage"""
    try:
        results = ml_service.load_all_models()
        loaded_count = sum(1 for success in results.values() if success)
        return {
            "message": f"Loaded {loaded_count} models from ImageKit",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models")
async def clear_models():
    """Clear all trained models from memory and ImageKit"""
    try:
        ml_service.clear_models()
        return {"message": "All models cleared successfully from memory and ImageKit"}
    except Exception as e:
        logger.error(f"Error clearing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
