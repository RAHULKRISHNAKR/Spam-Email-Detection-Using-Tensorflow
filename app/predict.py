import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Union, List, Dict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.preprocess import preprocess_text, load_tokenizer

class SpamPredictor:
    """
    A class for making spam predictions on new emails.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, max_len: int = 100, threshold: float = 0.3):
        """
        Initialize the SpamPredictor.
        
        Args:
            model_path (str): Path to the trained model
            tokenizer_path (str): Path to the saved tokenizer
            max_len (int): Maximum length of sequences
            threshold (float): Classification threshold (lowered to 0.3)
        """
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_len = max_len
        self.threshold = threshold
    
    def predict(self, texts: Union[str, List[str]]) -> Dict:
        """
        Make predictions on new text(s).
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to predict
            
        Returns:
            Dict: Dictionary with prediction results
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts_cleaned = [preprocess_text(text) for text in texts]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(texts_cleaned)
        
        # Pad sequences
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_len, padding='post', truncating='post'
        )
        
        # Make predictions
        probabilities = self.model.predict(padded_sequences).flatten()
        predictions = (probabilities > self.threshold).astype(int)
        
        # Format results
        results = []
        for i, (text, prob, pred) in enumerate(zip(texts, probabilities, predictions)):
            results.append({
                'text': text,
                'probability': float(prob),
                'prediction': int(pred),
                'label': 'SPAM' if pred == 1 else 'HAM',
                'confidence': float(prob) if pred == 1 else float(1 - prob)
            })
        
        return {
            'predictions': results,
            'num_spam': sum(predictions),
            'num_ham': len(predictions) - sum(predictions)
        }