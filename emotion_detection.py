from transformers import pipeline

# ---------------------------------
# Load Emotion Classification Model
# ---------------------------------
# This model outputs emotions like anger, fear, joy, sadness, etc.
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# ---------------------------------
# Emotion Mapping (Model → Project)
# ---------------------------------
EMOTION_MAP = {
    "anger": "Anger",
    "fear": "Fear",
    "sadness": "Sadness",
    "joy": "Hope",
    "neutral": "Trust",
    "surprise": "Trust",
    "disgust": "Anger"
}

# ---------------------------------
# Emotion Detection Function
# ---------------------------------
def detect_emotion(text: str):
    """
    Detect dominant emotion in climate-related text.

    Args:
        text (str): Input tweet/text

    Returns:
        dict: dominant emotion and probability
    """

    results = emotion_classifier(text)[0]

    # Aggregate scores into required emotion set
    emotion_scores = {
        "Fear": 0.0,
        "Anger": 0.0,
        "Hope": 0.0,
        "Sadness": 0.0,
        "Trust": 0.0
    }

    for item in results:
        label = item["label"].lower()
        score = item["score"]

        mapped_emotion = EMOTION_MAP.get(label)
        if mapped_emotion:
            emotion_scores[mapped_emotion] += score

    # Get dominant emotion
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = emotion_scores[dominant_emotion]

    return {
        "emotion": dominant_emotion,
        "confidence": round(confidence, 4)
    }
