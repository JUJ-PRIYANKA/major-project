import joblib
import numpy as np

# ---------------------------------
# Load Trained Random Forest Model
# ---------------------------------
MODEL_PATH = "models/rf_climate_sentiment.pkl"

rf_model = joblib.load(MODEL_PATH)

# ---------------------------------
# Class Labels (Fixed Order)
# ---------------------------------
CLASS_LABELS = ["Negative", "Neutral", "Positive"]

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_sentiment_rf(embeddings: np.ndarray):
    """
    Predict sentiment using trained Random Forest model.

    Args:
        embeddings (np.ndarray): ClimateBERT embeddings (shape: [n_samples, 768])

    Returns:
        dict: sentiment label and confidence score
    """

    # Predict class index
    pred_class_index = rf_model.predict(embeddings)[0]

    # Predict class probabilities
    probabilities = rf_model.predict_proba(embeddings)[0]

    # Get confidence score (max probability)
    confidence = float(np.max(probabilities))

    # Map index to sentiment label
    sentiment = CLASS_LABELS[pred_class_index]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }
