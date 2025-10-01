from pydantic import BaseModel, Field
from typing import List

class SensorData(BaseModel):
    Time: int
    Type: str
    Concentration: int
    sensor_readings: List[float] = Field(default_factory=list)
