from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference_pipeline import run_inference

# ---------------------------------
# FastAPI App Initialization
# ---------------------------------
app = FastAPI(
    title="Climate Change Sentiment Analysis API",
    description="Real-time inference API using ClimateBERT",
    version="1.0"
)

# ---------------------------------
# Pydantic Schemas
# ---------------------------------
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    misinformation_label: str
    emotion: str


# ---------------------------------
# Startup Event (Load Models Once)
# ---------------------------------
@app.on_event("startup")
def load_models():
    """
    This ensures all models are loaded once at startup.
    ClimateBERT, RF sentiment model, misinformation model,
    and emotion model are initialized via imports.
    """
    print("✅ Models loaded and ready for inference")


# ---------------------------------
# Real-Time Prediction Endpoint
# ---------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Perform real-time inference on a climate-related tweet.
    """

    if not request.text or request.text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    try:
        result = run_inference(request.text)

        return {
            "sentiment": result["sentiment"],
            "confidence": result["sentiment_confidence"],
            "misinformation_label": result["misinformation"],
            "emotion": result["emotion"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
