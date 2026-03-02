import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Dictionaries based on Software Engineering heuristics
MODAL_VERBS = {"shall", "must", "should", "could", "would", "may", "might"}
VAGUE_WORDS = {"fast", "secure", "reliable", "efficient", "robust", "flexible", 
               "easy", "quick", "seamless", "user-friendly", "intuitive", 
               "state-of-the-art", "modern", "scalable", "approximately", 
               "some", "many", "few", "several", "often", "usually"}
INCOMPLETE_MARKERS = {"tbd", "to be decided", "to be defined", "to be determined", "etc"}

def extract_features(text):
    """
    Extracts heuristic features from a raw SRS text sentence.
    Returns a dictionary of features.
    """
    text_lower = text.lower()
    tokens = set(re.findall(r'\b\w+\b', text_lower))
    
    features = {
        'modal_count': sum(1 for word in tokens if word in MODAL_VERBS),
        'has_numeric': 1 if re.search(r'\b\d+(\.\d+)?\b', text) else 0,
        'has_vague': 1 if any(word in tokens for word in VAGUE_WORDS) else 0,
        'has_incomplete_marker': 1 if any(marker in text_lower for marker in INCOMPLETE_MARKERS) else 0,
        'sentence_length': len(tokens)
    }
    return features

def get_vectorizer():
    """Returns a configured TF-IDF Vectorizer (unigrams + bigrams) with controlled feature count"""
    return TfidfVectorizer(
        max_features=1000,  # Controlled to prevent overfitting while retaining signal
        ngram_range=(1, 2),
        stop_words='english'
    )
