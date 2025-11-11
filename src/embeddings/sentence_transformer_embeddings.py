"""
Sentence Transformer embedding generator
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import SENTENCE_TRANSFORMER_CONFIG
from src.utils.logger_utils import setup_logger

logger = setup_logger('sentence_transformer_embeddings')

class SentenceTransformerEmbeddings:
    """Class to generate Sentence Transformer embeddings"""
    
    def __init__(self, config=SENTENCE_TRANSFORMER_CONFIG):
        """
        Initialize Sentence Transformer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['model_name']
        self.batch_size = config['batch_size']
        self.model = None
        
    def load_model(self):
        """Load pre-trained Sentence Transformer model"""
        logger.info(f"Loading Sentence Transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")
        
    def transform(self, texts):
        """
        Transform texts to embeddings
        
        Args:
            texts: List of text strings (not tokens)
        
        Returns:
            embeddings: Array of sentence embeddings
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
