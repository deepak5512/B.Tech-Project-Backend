from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
sensor_collection = db["sensors"]
ml_evaluations_collection = db["ml_evaluations"]
ml_predictions_collection = db["ml_predictions"]
