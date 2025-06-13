import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import time
from typing import Dict, Tuple, Any, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_spam_dataset, get_data_info
from data.preprocess import preprocess_data, save_tokenizer
from models.spam_classifier import create_lstm_model, create_cnn_model, get_callbacks

def train_model(
    data_path: str,
    model_type: str = 'lstm',
    max_features: int = 10000,
    max_len: int = 100,
    embedding_dim: int = 100,
    batch_size: int = 64,
    epochs: int = 10,
    output_dir: str = 'output',
    random_state: int = 42
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train a spam classification model.
    
    Args:
        data_path (str): Path to the dataset
        model_type (str): Type of model to use ('lstm' or 'cnn')
        max_features (int): Maximum number of words to keep in vocabulary
        max_len (int): Maximum length of sequences
        embedding_dim (int): Dimension of embeddings
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        output_dir (str): Directory to save model and outputs
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: (model, history)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    tokenizer_dir = os.path.join(output_dir, 'tokenizer')
    logs_dir = os.path.join(output_dir, 'logs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Load dataset
    print("Loading dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_spam_dataset(
        data_path=data_path,
        test_size=0.2,
        val_size=0.2,
        random_state=random_state
    )
    
    # Print dataset info
    get_data_info(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_seq, X_val_seq, X_test_seq, tokenizer = preprocess_data(
        X_train, X_val, X_test,
        max_features=max_features,
        max_len=max_len
    )
    
    # Save tokenizer
    save_tokenizer(tokenizer, os.path.join(tokenizer_dir, 'tokenizer.pickle'))
    
    # Create model
    print(f"Creating {model_type.upper()} model...")
    if model_type.lower() == 'lstm':
        model = create_lstm_model(
            vocab_size=max_features,
            embedding_dim=embedding_dim,
            max_length=max_len
        )
    elif model_type.lower() == 'cnn':
        model = create_cnn_model(
            vocab_size=max_features,
            embedding_dim=embedding_dim,
            max_length=max_len
        )
    else:
        raise ValueError("Model type must be 'lstm' or 'cnn'")
    
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks(
        checkpoint_dir=checkpoint_dir,
        tensorboard_log_dir=logs_dir
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(output_dir, 'spam_classifier_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    
    return model, history

def plot_training_history(history: Dict[str, list], output_path: str) -> None:
    """
    Plot training history and save the figure.
    
    Args:
        history: Training history
        output_path: Path to save the plot
    """
    # Plot training & validation accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train spam classification model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn'], help='Type of model to use')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum number of words in vocabulary')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of sequences')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and outputs')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_type=args.model_type,
        max_features=args.max_features,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

if __name__ == '__main__':
    main()