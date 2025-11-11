"""
Configuration file for the sentiment analysis project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, 
                  MODELS_DIR, EMBEDDINGS_DIR, METRICS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data parameters
DATA_FILE = RAW_DATA_DIR / "stock_news.csv"
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Text preprocessing parameters
MAX_VOCAB_SIZE = 50000
MIN_WORD_FREQUENCY = 2
MAX_SEQUENCE_LENGTH = 512

# Embedding parameters
WORD2VEC_CONFIG = {
    'vector_size': 300,
    'window': 5,
    'min_count': 2,
    'workers': 4,
    'epochs': 10,
    'sg': 1  # Skip-gram
}

GLOVE_CONFIG = {
    'embedding_dim': 300,
    'glove_file': 'glove.6B.300d.txt'  # Download from Stanford NLP
}

FASTTEXT_CONFIG = {
    'dim': 300,
    'epoch': 10,
    'lr': 0.1,
    'wordNgrams': 2,
    'minCount': 2
}

SENTENCE_TRANSFORMER_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Fast and efficient
    'batch_size': 32
}

# Random Forest hyperparameters for grid search
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Model evaluation parameters
CV_FOLDS = 5
SCORING_METRICS = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

# Label mapping
LABEL_MAPPING = {
    -1: 'Negative',
    0: 'Neutral',
    1: 'Positive'
}
