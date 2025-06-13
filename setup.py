from setuptools import setup, find_packages

setup(
    name="spam_email_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.9.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "nltk>=3.6.0",
    ],
)