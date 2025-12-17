import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ---------------------------------
# Model Configuration
# ---------------------------------
MODEL_NAME = "climatebert/distilroberta-base-climate-f"

# ---------------------------------
# Device Selection (CPU/GPU)
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# Load Tokenizer & Model (once)
# ---------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()  # Important for inference optimization

# ---------------------------------
# Disable gradients for inference
# ---------------------------------
torch.set_grad_enabled(False)

# ---------------------------------
# Embedding Generation Function
# ---------------------------------
def generate_climatebert_embeddings(
    texts,
    max_length: int = 128
):
    """
    Convert text(s) into ClimateBERT embeddings.

    Args:
        texts (str or list[str]): Input text or list of texts
        max_length (int): Max token length

    Returns:
        np.ndarray: Embeddings compatible with scikit-learn
    """

    # Ensure input is list
    if isinstance(texts, str):
        texts = [texts]

    # Tokenization
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Move tensors to device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass
    outputs = model(**encoded)

    # Use CLS token embedding (sentence representation)
    embeddings = outputs.last_hidden_state[:, 0, :]

    # Convert to NumPy for scikit-learn compatibility
    embeddings = embeddings.cpu().numpy()

    return embeddings
