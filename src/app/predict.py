import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Union, List, Dict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import preprocess_text, load_tokenizer

class SpamPredictor:
    """
    A class for making spam predictions on new emails.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, max_len: int = 100, threshold: float = 0.5):
        """
        Initialize the SpamPredictor.
        
        Args:
            model_path (str): Path to the trained model
            tokenizer_path (str): Path to the saved tokenizer
            max_len (int): Maximum length of sequences
            threshold (float): Classification threshold
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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict spam or ham for input text')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the saved tokenizer')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='Path to file with texts to classify (one per line)')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of sequences')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    # Ensure either text or file is provided
    if args.text is None and args.file is None:
        parser.error("Either --text or --file must be provided")
    
    # Initialize predictor
    predictor = SpamPredictor(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        max_len=args.max_len,
        threshold=args.threshold
    )
    
    # Process text or file
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Make predictions
    results = predictor.predict(texts)
    
    # Print results
    for i, prediction in enumerate(results['predictions']):
        confidence_pct = prediction['confidence'] * 100
        print(f"Text {i+1}: {prediction['label']} (confidence: {confidence_pct:.2f}%)")
        print(f"  '{prediction['text'][:100]}{'...' if len(prediction['text']) > 100 else ''}'")
        print()
    
    print(f"Summary: {results['num_spam']} spam, {results['num_ham']} ham")

if __name__ == '__main__':
    main()