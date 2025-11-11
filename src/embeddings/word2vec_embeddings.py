"""
Word2Vec embedding generator
"""
import numpy as np
from gensim.models import Word2Vec
import joblib
from pathlib import Path
from src.utils.config import WORD2VEC_CONFIG, EMBEDDINGS_DIR
from src.utils.logger_utils import setup_logger

logger = setup_logger('word2vec_embeddings')

class Word2VecEmbeddings:
    """Class to generate Word2Vec embeddings"""
    
    def __init__(self, config=WORD2VEC_CONFIG):
        """
        Initialize Word2Vec embeddings
        
        Args:
            config: Configuration dictionary for Word2Vec
        """
        self.config = config
        self.model = None
        self.embedding_dim = config['vector_size']
        
    def train(self, tokens_list):
        """
        Train Word2Vec model
        
        Args:
            tokens_list: List of tokenized sentences
        """
        logger.info("Training Word2Vec model...")
        logger.info(f"Configuration: {self.config}")
        
        self.model = Word2Vec(
            sentences=tokens_list,
            **self.config
        )
        
        logger.info(f"Word2Vec model trained. Vocabulary size: {len(self.model.wv)}")
        
    def save_model(self, filename='word2vec_model.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        filepath = EMBEDDINGS_DIR / filename
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filename='word2vec_model.pkl'):
        """Load pre-trained model"""
        filepath = EMBEDDINGS_DIR / filename
        self.model = joblib.load(filepath)
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
        
        # Get word vectors for tokens in vocabulary
        word_vectors = []
        for token in tokens:
            if token in self.model.wv:
                word_vectors.append(self.model.wv[token])
        
        # Average word vectors
        if len(word_vectors) > 0:
            embedding = np.mean(word_vectors, axis=0)
        else:
            # Return zero vector if no words in vocabulary
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
