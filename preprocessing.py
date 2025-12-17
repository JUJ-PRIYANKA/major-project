import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------
# Download required NLTK resources
# (run once)
# ---------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------------
# Initialize tools
# ---------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------------
# Reusable Preprocessing Function
# ---------------------------------
def preprocess_tweet(text: str) -> str:
    """
    Preprocess a climate-related tweet:
    - Lowercase
    - Remove URLs, emojis, symbols
    - Tokenization
    - Stopword removal
    - Lemmatization

    Args:
        text (str): Raw tweet text

    Returns:
        str: Cleaned and processed text
    """

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # 4. Remove emojis & special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Stopword removal & Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # 7. Join tokens back to string
    cleaned_text = " ".join(cleaned_tokens)

    return cleaned_text
