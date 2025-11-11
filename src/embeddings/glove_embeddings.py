"""
GloVe embedding generator
"""
import numpy as np
from pathlib import Path
import joblib
from src.utils.config import GLOVE_CONFIG, EMBEDDINGS_DIR
from src.utils.logger_utils import setup_logger

logger = setup_logger('glove_embeddings')

class GloVeEmbeddings:
    """Class to use pre-trained GloVe embeddings"""
    
    def __init__(self, config=GLOVE_CONFIG):
        """
        Initialize GloVe embeddings
        
        Args:
            config: Configuration dictionary for GloVe
        """
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.embeddings_index = {}
        
    def load_glove_embeddings(self, glove_path=None):
        """
        Load pre-trained GloVe embeddings
        
        Args:
            glove_path: Path to GloVe file (default from config)
        """
        if glove_path is None:
            glove_path = EMBEDDINGS_DIR / self.config['glove_file']
        
        if not Path(glove_path).exists():
            logger.error(f"GloVe file not found at {glove_path}")
            logger.info("Please download GloVe embeddings from:")
            logger.info("https://nlp.stanford.edu/projects/glove/")
            logger.info(f"And place the file in: {EMBEDDINGS_DIR}")
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")
        
        logger.info(f"Loading GloVe embeddings from {glove_path}...")
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = vector
        
        logger.info(f"Loaded {len(self.embeddings_index)} word vectors")
        
    def get_document_embedding(self, tokens):
        """
        Get document embedding by averaging GloVe word vectors
        
        Args:
            tokens: List of tokens
        
        Returns:
            embedding: Document embedding vector
        """
        word_vectors = []
        for token in tokens:
            if token in self.embeddings_index:
                word_vectors.append(self.embeddings_index[token])
        
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
        if len(self.embeddings_index) == 0:
            raise ValueError("GloVe embeddings not loaded. Call load_glove_embeddings() first.")
        
        logger.info(f"Generating embeddings for {len(tokens_list)} documents...")
        
        embeddings = np.array([
            self.get_document_embedding(tokens) for tokens in tokens_list
        ])
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
