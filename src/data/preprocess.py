from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Union

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text: str) -> str:
    """
    Clean the text by removing special characters, numbers, and extra spaces.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Preprocess the text by cleaning, removing stopwords and lemmatizing.
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
        
    Returns:
        str: Preprocessed text
    """
    # Clean text
    text = clean_text(text)
    
    # Simple tokenization (avoid using nltk.word_tokenize)
    words = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def preprocess_data(X_train: List[str], X_val: List[str], X_test: List[str], 
                    max_features: int = 10000, max_len: int = 100) -> Tuple:
    """
    Preprocess the data by cleaning, tokenizing, and padding sequences.
    
    Args:
        X_train (List[str]): Training text data
        X_val (List[str]): Validation text data
        X_test (List[str]): Test text data
        max_features (int): Maximum number of words to keep in the vocabulary
        max_len (int): Maximum length of sequences
        
    Returns:
        Tuple: (X_train_seq, X_val_seq, X_test_seq, tokenizer)
    """
    # Preprocess text
    X_train_clean = [preprocess_text(text) for text in X_train]
    X_val_clean = [preprocess_text(text) for text in X_val]
    X_test_clean = [preprocess_text(text) for text in X_test]
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_clean)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
    X_val_seq = tokenizer.texts_to_sequences(X_val_clean)
    X_test_seq = tokenizer.texts_to_sequences(X_test_clean)
    
    # Pad sequences
    X_train_seq = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_seq = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_seq = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    return X_train_seq, X_val_seq, X_test_seq, tokenizer

def save_tokenizer(tokenizer: Tokenizer, output_path: str) -> None:
    """
    Save the tokenizer to a pickle file.
    
    Args:
        tokenizer (Tokenizer): Tokenizer to save
        output_path (str): Path to save the tokenizer
    """
    with open(output_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Tokenizer saved to {output_path}")

def load_tokenizer(input_path: str) -> Tokenizer:
    """
    Load the tokenizer from a pickle file.
    
    Args:
        input_path (str): Path to load the tokenizer from
        
    Returns:
        Tokenizer: Loaded tokenizer
    """
    with open(input_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print(f"Tokenizer loaded from {input_path}")
    return tokenizer