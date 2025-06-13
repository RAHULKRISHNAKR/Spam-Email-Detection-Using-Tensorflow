import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, Bidirectional, 
    Dropout, Conv1D, GlobalMaxPooling1D, Input,
    SpatialDropout1D, GlobalAveragePooling1D, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
import os
from typing import Dict, Any, Union, Tuple, Optional

def create_lstm_model(
    vocab_size: int, 
    embedding_dim: int = 100, 
    max_length: int = 100,
    lstm_units: int = 128,
    dropout_rate: float = 0.2
) -> Model:
    """
    Create a simple LSTM model for text classification.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding
        max_length (int): Maximum length of sequences
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: Compiled Keras model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        SpatialDropout1D(dropout_rate),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()]
    )
    
    return model

def create_cnn_model(
    vocab_size: int,
    embedding_dim: int = 100,
    max_length: int = 100,
    filters: int = 128,
    kernel_sizes: list = [3, 4, 5],
    dropout_rate: float = 0.2
) -> Model:
    """
    Create a CNN model for text classification.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding
        max_length (int): Maximum length of sequences
        filters (int): Number of filters for each convolution
        kernel_sizes (list): List of kernel sizes for convolutions
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(max_length,))
    
    # Embedding layer
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
    dropout_embed = SpatialDropout1D(dropout_rate)(embedding)
    
    # Parallel convolution layers
    conv_blocks = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(dropout_embed)
        max_pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(max_pool)
    
    # Concatenate convolution outputs
    concat = concatenate(conv_blocks) if len(kernel_sizes) > 1 else conv_blocks[0]
    
    # Dense layers
    dropout = Dropout(dropout_rate)(concat)
    dense = Dense(100, activation='relu')(dropout)
    dropout2 = Dropout(dropout_rate)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy',
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()]
    )
    
    return model

def get_callbacks(
    checkpoint_dir: str,
    tensorboard_log_dir: str,
    patience_early: int = 5,
    patience_lr: int = 3
) -> list:
    """
    Create callbacks for model training.
    
    Args:
        checkpoint_dir (str): Directory to save model checkpoints
        tensorboard_log_dir (str): Directory for tensorboard logs
        patience_early (int): Patience for early stopping
        patience_lr (int): Patience for learning rate reduction
        
    Returns:
        list: List of callbacks
    """
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "model_best.h5")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks