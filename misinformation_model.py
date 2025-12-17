import joblib
import numpy as np

# ---------------------------------
# Load Trained Misinformation Model
# ---------------------------------
MODEL_PATH = "models/misinformation_classifier.pkl"

misinfo_model = joblib.load(MODEL_PATH)

# ---------------------------------
# Class Labels
# ---------------------------------
CLASS_LABELS = ["Scientific", "Misinformation"]

# ---------------------------------
# Prediction Function
# ---------------------------------
def detect_misinformation(embeddings: np.ndarray):
    """
    Detect whether the given text is scientific or misinformation.

    Args:
        embeddings (np.ndarray): ClimateBERT embeddings (shape: [n_samples, 768])

    Returns:
        dict: misinformation label and confidence
    """

    # Predict class index
    pred_class_index = misinfo_model.predict(embeddings)[0]

    # Predict class probabilities
    probabilities = misinfo_model.predict_proba(embeddings)[0]

    # Get confidence score
    confidence = float(np.max(probabilities))

    # Map index to label
    label = CLASS_LABELS[pred_class_index]

    return {
        "misinformation": label,
        "confidence": round(confidence, 4)
    }
