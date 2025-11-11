"""
FastText embedding generator
"""
import numpy as np
import fasttext
import tempfile
from pathlib import Path
import joblib
from src.utils.config import FASTTEXT_CONFIG, EMBEDDINGS_DIR
from src.utils.logger_utils import setup_logger

logger = setup_logger('fasttext_embeddings')

class FastTextEmbeddings:
    """Class to generate FastText embeddings"""
    
    def __init__(self, config=FASTTEXT_CONFIG):
        """
        Initialize FastText embeddings
        
        Args:
            config: Configuration dictionary for FastText
        """
        self.config = config
        self.model = None
        self.embedding_dim = config['dim']
        
    def train(self, tokens_list):
        """
        Train FastText model
        
        Args:
            tokens_list: List of tokenized sentences
        """
        logger.info("Training FastText model...")
        logger.info(f"Configuration: {self.config}")
        
        # FastText requires a text file as input
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
            for tokens in tokens_list:
                f.write(' '.join(tokens) + '\n')
        
        # Train model
        self.model = fasttext.train_unsupervised(
            temp_file,
            model='skipgram',
            **self.config
        )
        
        # Clean up temp file
        Path(temp_file).unlink()
        
        logger.info(f"FastText model trained. Vocabulary size: {len(self.model.words)}")
        
    def save_model(self, filename='fasttext_model.bin'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        filepath = EMBEDDINGS_DIR / filename
        self.model.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filename='fasttext_model.bin'):
        """Load pre-trained model"""
        filepath = EMBEDDINGS_DIR / filename
        self.model = fasttext.load_model(str(filepath))
        logger.info(f"Model loaded from {filepath}")
        
    def get_document_embedding(self, tokens):
        """
        Get document embedding by averaging word vectors
        
        Args:
            tokens: List of tokens
        
        Returns:
            embedding: Document embedding vector
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Get word vectors
        word_vectors = [self.model.get_word_vector(token) for token in tokens]
        
        if len(word_vectors) > 0:
            embedding = np.mean(word_vectors, axis=0)
        else:
            embedding = np.zeros(self.embedding_dim)
        
        return embedding
    
    def transform(self, tokens_list):
        """
        Transform list of tokenized documents to embeddings
        
        Args:
            tokens_list: List of tokenized sentences
        
        Returns:
            embeddings: Array of document embeddings
        """
        logger.info(f"Generating embeddings for {len(tokens_list)} documents...")
        
        embeddings = np.array([
            self.get_document_embedding(tokens) for tokens in tokens_list
        ])
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
