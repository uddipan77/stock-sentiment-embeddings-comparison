"""
Data preprocessing utilities for text cleaning
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from tqdm import tqdm
from src.utils.logger_utils import setup_logger

logger = setup_logger('data_preprocessing')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Class to handle text preprocessing"""
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """
        Initialize TextPreprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def clean_text(self, text):
        """
        Clean individual text string
        
        Args:
            text: Input text string
        
        Returns:
            cleaned_text: Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_process(self, text):
        """
        Tokenize and process text
        
        Args:
            text: Input text string
        
        Returns:
            tokens: List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize if specified
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove short tokens (length < 2)
        tokens = [word for word in tokens if len(word) > 2]
        
        return tokens
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text
        
        Args:
            text: Input text string
        
        Returns:
            processed_text: Processed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize and process
        tokens = self.tokenize_and_process(text)
        
        # Join tokens back into string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df, text_column='News'):
        """
        Preprocess all texts in a dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of the column containing text
        
        Returns:
            df: Dataframe with added 'processed_text' column
        """
        logger.info(f"Preprocessing {len(df)} texts...")
        
        tqdm.pandas(desc="Processing texts")
        df['processed_text'] = df[text_column].progress_apply(self.preprocess_text)
        
        # Remove empty processed texts
        original_len = len(df)
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
        removed = original_len - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} empty texts after preprocessing")
        
        logger.info("Preprocessing completed")
        return df
    
    def get_tokens_list(self, df, text_column='processed_text'):
        """
        Get list of tokenized texts for training embeddings
        
        Args:
            df: Input dataframe
            text_column: Name of the column containing processed text
        
        Returns:
            tokens_list: List of token lists
        """
        tokens_list = df[text_column].apply(lambda x: x.split()).tolist()
        return tokens_list
