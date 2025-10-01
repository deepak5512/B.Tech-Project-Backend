from fastapi import APIRouter
from models import SensorData
from typing import List
import crud

router = APIRouter(prefix="/sensors", tags=["Sensors"])

@router.post("/")
def add_sensor(data: SensorData):
    return crud.insert_sensor(data)

@router.post("/batch")
def add_sensors_batch(data: dict):
    sensor_data_list = data.get("data", [])
    results = []
    success_count = 0
    error_count = 0
    
    for sensor_data in sensor_data_list:
        try:
            # Convert dict to SensorData model
            sensor = SensorData(**sensor_data)
            result = crud.insert_sensor(sensor)
            results.append({"success": True, "id": result["id"]})
            success_count += 1
        except Exception as e:
            results.append({"success": False, "error": str(e)})
            error_count += 1
    
    return {
        "message": f"Batch upload completed. {success_count} successful, {error_count} failed.",
        "success_count": success_count,
        "error_count": error_count,
        "results": results
    }

@router.get("/")
def get_sensors():
    return crud.fetch_all_sensors()
