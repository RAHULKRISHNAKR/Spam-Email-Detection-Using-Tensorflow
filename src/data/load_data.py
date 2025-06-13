import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_spam_dataset(data_path, test_size=0.2, val_size=0.2, random_state=42):
    """
    Load the spam dataset from a CSV file and split it into train, validation, and test sets.
    
    Args:
        data_path (str): Path to the CSV file containing the spam dataset
        test_size (float): Proportion of the dataset to include in the test split
        val_size (float): Proportion of the training dataset to include in the validation split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Check expected columns - adjust column names based on your dataset
    if 'v1' not in df.columns or 'v2' not in df.columns:
        raise ValueError("Dataset does not have the expected columns. Expected 'v1' for labels and 'v2' for messages.")
    
    # Rename columns for clarity
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split into features and target
    X = df['message'].values
    y = df['label'].values
    
    # First split: training + validation, and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size),  # Adjust validation size
        random_state=random_state,
        stratify=y_train_val
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_data_info(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Print information about the dataset splits.
    """
    print(f"Training set: {len(X_train)} samples")
    print(f"  - Spam: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"  - Ham: {len(y_train) - sum(y_train)} ({(1 - sum(y_train)/len(y_train))*100:.1f}%)")
    
    print(f"Validation set: {len(X_val)} samples")
    print(f"  - Spam: {sum(y_val)} ({sum(y_val)/len(y_val)*100:.1f}%)")
    print(f"  - Ham: {len(y_val) - sum(y_val)} ({(1 - sum(y_val)/len(y_val))*100:.1f}%)")
    
    print(f"Test set: {len(X_test)} samples")
    print(f"  - Spam: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"  - Ham: {len(y_test) - sum(y_test)} ({(1 - sum(y_test)/len(y_test))*100:.1f}%)")