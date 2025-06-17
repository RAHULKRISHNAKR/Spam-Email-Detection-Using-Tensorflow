import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    precision_recall_curve, auc, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import sys
from typing import Dict, Tuple, Any, List

# Fix the import path to properly include the project root directory
# Change this line:
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# To this:
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Now the imports will work
from src.data.load_data import load_spam_dataset
from src.data.preprocess import preprocess_data, load_tokenizer, preprocess_text

def evaluate_model(
    model_path: str,
    data_path: str,
    tokenizer_path: str,
    max_len: int = 100,
    output_dir: str = 'evaluation',
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the trained model on the test set.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the dataset
        tokenizer_path (str): Path to the saved tokenizer
        max_len (int): Maximum length of sequences
        output_dir (str): Directory to save evaluation results
        threshold (float): Classification threshold
        
    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load dataset
    print("Loading dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_spam_dataset(
        data_path=data_path,
        test_size=0.2,
        val_size=0.2
    )
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Preprocess test data
    print("Preprocessing test data...")
    X_test_clean = [preprocess_text(text) for text in X_test]
    X_test_seq = tokenizer.texts_to_sequences(X_test_clean)
    X_test_seq = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Predict on test data
    print("Making predictions...")
    y_pred_proba = model.predict(X_test_seq)
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Create and save confusion matrix
    plot_confusion_matrix(y_test, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot and save ROC curve
    plot_roc_curve(y_test, y_pred_proba, os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot and save precision-recall curve
    plot_precision_recall_curve(y_test, y_pred_proba, os.path.join(output_dir, 'pr_curve.png'))
    
    # Find examples of correct and incorrect predictions
    find_prediction_examples(X_test, y_test, y_pred, y_pred_proba, 
                             os.path.join(output_dir, 'prediction_examples.csv'))
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        output_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        output_path (str): Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"ROC curve saved to {output_path}")

def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str) -> None:
    """
    Plot and save precision-recall curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        output_path (str): Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Precision-Recall curve saved to {output_path}")

def find_prediction_examples(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str,
    num_examples: int = 5
) -> None:
    """
    Find examples of correct and incorrect predictions and save them to a CSV file.
    
    Args:
        X_test (np.ndarray): Test text data
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities
        output_path (str): Path to save the examples
        num_examples (int): Number of examples to find for each category
    """
    # Flatten arrays
    y_pred_proba = y_pred_proba.flatten()
    
    # Create dataframe with all test examples
    df = pd.DataFrame({
        'text': X_test,
        'true_label': y_test,
        'predicted_label': y_pred,
        'confidence': y_pred_proba,
        'correct': y_test == y_pred
    })
    
    # Add category labels
    df['true_category'] = df['true_label'].map({0: 'Ham', 1: 'Spam'})
    df['predicted_category'] = df['predicted_label'].map({0: 'Ham', 1: 'Spam'})
    
    # Find examples of each case (true positives, false positives, etc.)
    true_positives = df[(df['true_label'] == 1) & (df['predicted_label'] == 1)].sort_values(by='confidence', ascending=False).head(num_examples)
    true_negatives = df[(df['true_label'] == 0) & (df['predicted_label'] == 0)].sort_values(by='confidence', ascending=True).head(num_examples)
    false_positives = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)].sort_values(by='confidence', ascending=False).head(num_examples)
    false_negatives = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)].sort_values(by='confidence', ascending=True).head(num_examples)
    
    # Combine examples
    examples = pd.concat([
        true_positives.assign(case='True Positive'),
        true_negatives.assign(case='True Negative'),
        false_positives.assign(case='False Positive'),
        false_negatives.assign(case='False Negative')
    ])
    
    # Save to CSV
    examples.to_csv(output_path, index=False)
    print(f"Prediction examples saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate spam classification model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the saved tokenizer')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of sequences')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        max_len=args.max_len,
        output_dir=args.output_dir,
        threshold=args.threshold
    )

if __name__ == '__main__':
    main()