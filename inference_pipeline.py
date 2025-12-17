"""
Unified Inference Pipeline for Climate Change Sentiment Analysis

Steps:
1. Preprocessing
2. ClimateBERT Embedding Generation
3. Sentiment Prediction (RF)
4. Misinformation Detection
5. Emotion Detection
"""

# ---------------------------------
# Imports
# ---------------------------------
from preprocessing import preprocess_tweet
from climatebert_embeddings import generate_climatebert_embeddings
from rf_sentiment_model import predict_sentiment_rf
from misinformation_model import detect_misinformation
from emotion_detection import detect_emotion


# ---------------------------------
# Unified Inference Function
# ---------------------------------
def run_inference(raw_text: str) -> dict:
    """
    Run complete inference pipeline on a raw tweet.

    Args:
        raw_text (str): Raw climate-related tweet text

    Returns:
        dict: JSON-style response with sentiment, confidence,
              misinformation label, and emotion
    """

    if not raw_text or raw_text.strip() == "":
        raise ValueError("Input text is empty")

    # 1. Preprocessing
    cleaned_text = preprocess_tweet(raw_text)

    # 2. Generate ClimateBERT embeddings
    embeddings = generate_climatebert_embeddings(cleaned_text)

    # 3. Sentiment Prediction
    sentiment_result = predict_sentiment_rf(embeddings)

    # 4. Misinformation Detection
    misinfo_result = detect_misinformation(embeddings)

    # 5. Emotion Detection (uses raw or cleaned text)
    emotion_result = detect_emotion(raw_text)

    # ---------------------------------
    # Final JSON Response
    # ---------------------------------
    response = {
        "input_text": raw_text,
        "processed_text": cleaned_text,

        "sentiment": sentiment_result["sentiment"],
        "sentiment_confidence": sentiment_result["confidence"],

        "misinformation": misinfo_result["misinformation"],
        "misinformation_confidence": misinfo_result["confidence"],

        "emotion": emotion_result["emotion"],
        "emotion_confidence": emotion_result["confidence"]
    }

    return response
