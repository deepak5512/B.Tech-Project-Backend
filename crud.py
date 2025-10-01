from database import sensor_collection, ml_evaluations_collection, ml_predictions_collection
from models import SensorData
from ml_models import ModelEvaluation, PredictionResult
from bson import ObjectId

def sensor_helper(sensor) -> dict:
    return {
        "id": str(sensor["_id"]),
        "Time": sensor["Time"],
        "Type": sensor["Type"],
        "Concentration": sensor["Concentration"],
        "sensor_readings": sensor["sensor_readings"],
    }

def insert_sensor(data: SensorData):
    result = sensor_collection.insert_one(data.dict())
    return {"id": str(result.inserted_id)}

def fetch_all_sensors():
    sensors = []
    for sensor in sensor_collection.find():
        sensors.append(sensor_helper(sensor))
    return sensors

# ML-related CRUD operations
def save_model_evaluation(evaluation: ModelEvaluation):
    """Save model evaluation to database"""
    evaluation_dict = evaluation.dict()
    result = ml_evaluations_collection.insert_one(evaluation_dict)
    return {"id": str(result.inserted_id)}

def get_all_evaluations():
    """Get all model evaluations"""
    evaluations = []
    for evaluation in ml_evaluations_collection.find().sort("created_at", -1):
        evaluation["id"] = str(evaluation["_id"])
        del evaluation["_id"]
        evaluations.append(evaluation)
    return evaluations

def get_model_evaluations(model_name: str):
    """Get evaluations for a specific model"""
    evaluations = []
    for evaluation in ml_evaluations_collection.find({"model_name": model_name}).sort("created_at", -1):
        evaluation["id"] = str(evaluation["_id"])
        del evaluation["_id"]
        evaluations.append(evaluation)
    return evaluations

def save_prediction_result(prediction: PredictionResult):
    """Save prediction result to database"""
    prediction_dict = prediction.dict()
    result = ml_predictions_collection.insert_one(prediction_dict)
    return {"id": str(result.inserted_id)}

def get_all_predictions():
    """Get all prediction results"""
    predictions = []
    for prediction in ml_predictions_collection.find().sort("created_at", -1):
        prediction["id"] = str(prediction["_id"])
        del prediction["_id"]
        predictions.append(prediction)
    return predictions
