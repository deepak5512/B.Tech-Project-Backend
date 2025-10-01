import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import sensor_routes, ml_routes
from ml_service import ml_service

app = FastAPI(
    title="Sensor Data API",
    summary="An API to serve sensor readings from a MongoDB database.",
)

@app.on_event("startup")
async def startup_event():
    """Load models from ImageKit on startup"""
    try:
        print("Loading models from ImageKit...")
        results = ml_service.load_all_models()
        loaded_count = sum(1 for success in results.values() if success)
        print(f"Loaded {loaded_count} models from ImageKit")
    except Exception as e:
        print(f"Error loading models on startup: {str(e)}")

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sensor_routes.router)
app.include_router(ml_routes.router)

@app.get("/")
def root():
    return {"message": "Welcome to Sensor Data API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
