from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ModelEvaluation(BaseModel):
    model_name: str
    model_type: str  # 'classification' or 'regression'
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    training_time: float
    created_at: datetime = Field(default_factory=datetime.now)

class PredictionResult(BaseModel):
    model_name: str
    model_type: str
    predicted_type: Optional[str] = None
    predicted_concentration: Optional[float] = None
    confidence: Optional[float] = None
    input_data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

class TrainingJob(BaseModel):
    job_id: str
    status: str  # 'running', 'completed', 'failed'
    models_trained: List[str]
    evaluation_results: List[ModelEvaluation]
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
