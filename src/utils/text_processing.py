import re
import string
import numpy as np
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

def get_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Get basic statistics about a list of texts.
    
    Args:
        texts (List[str]): List of text samples
        
    Returns:
        Dict[str, Any]: Dictionary with text statistics
    """
    # Calculate length of each text
    text_lengths = [len(text.split()) for text in texts]
    
    stats = {
        'count': len(texts),
        'avg_length': np.mean(text_lengths),
        'median_length': np.median(text_lengths),
        'min_length': min(text_lengths),
        'max_length': max(text_lengths),
        'std_length': np.std(text_lengths),
    }
    
    return stats

def plot_text_length_distribution(texts: List[str], title: str = "Text Length Distribution") -> None:
    """
    Plot the distribution of text lengths.
    
    Args:
        texts (List[str]): List of text samples
        title (str): Title for the plot
    """
    text_lengths = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_word_cloud(texts: List[str], title: str = "Word Cloud") -> None:
    """
    Generate and plot a word cloud from the texts.
    
    Args:
        texts (List[str]): List of text samples
        title (str): Title for the plot
    """
    # Combine all texts
    text_combined = ' '.join(texts)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text_combined)
    
    # Plot word cloud
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def get_most_common_words(texts: List[str], n: int = 20) -> List[tuple]:
    """
    Get the most common words in the texts.
    
    Args:
        texts (List[str]): List of text samples
        n (int): Number of most common words to return
        
    Returns:
        List[tuple]: List of (word, count) tuples
    """
    # Combine all texts and split into words
    words = ' '.join(texts).lower().split()
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get most common words
    most_common = word_counts.most_common(n)
    
    return most_common

def plot_most_common_words(texts: List[str], n: int = 20, title: str = "Most Common Words") -> None:
    """
    Plot the most common words in the texts.
    
    Args:
        texts (List[str]): List of text samples
        n (int): Number of most common words to plot
        title (str): Title for the plot
    """
    most_common = get_most_common_words(texts, n)
    words, counts = zip(*most_common)
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(words)), counts, align='center')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()