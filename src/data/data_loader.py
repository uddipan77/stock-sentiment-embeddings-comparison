"""
Data loading utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.config import *
from src.utils.logger_utils import setup_logger

logger = setup_logger('data_loader')

class DataLoader:
    """Class to handle data loading and splitting"""
    
    def __init__(self, data_path=DATA_FILE):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            
            # Display basic info
            logger.info(f"\nDataset Info:")
            logger.info(f"Total samples: {len(self.df)}")
            logger.info(f"Label distribution:\n{self.df['Label'].value_counts()}")
            
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self):
        """Validate data structure and content"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check required columns
        required_columns = ['Date', 'News', 'Label']
        missing_columns = set(required_columns) - set(self.df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_news = self.df['News'].isna().sum()
        missing_labels = self.df['Label'].isna().sum()
        
        if missing_news > 0:
            logger.warning(f"Found {missing_news} missing values in 'News' column")
        if missing_labels > 0:
            logger.warning(f"Found {missing_labels} missing values in 'Label' column")
        
        # Check label values
        unique_labels = self.df['Label'].unique()
        logger.info(f"Unique labels: {unique_labels}")
        
        return True
    
    def split_data(self, train_size=TRAIN_SIZE, val_size=VAL_SIZE, 
                   test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """
        Split data into train, validation, and test sets
        
        Args:
            train_size: Proportion of training data
            val_size: Proportion of validation data
            test_size: Proportion of test data
            random_state: Random state for reproducibility
        
        Returns:
            train_df, val_df, test_df: Split dataframes
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Verify splits sum to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Train, val, and test sizes must sum to 1"
        
        logger.info(f"Splitting data: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # First split: train+val vs test
        temp_size = train_size + val_size
        train_val_df, test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.df['Label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / temp_size
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['Label']
        )
        
        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        # Save splits
        self._save_splits(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _save_splits(self, train_df, val_df, test_df):
        """Save data splits to CSV files"""
        try:
            train_path = PROCESSED_DATA_DIR / "train.csv"
            val_path = PROCESSED_DATA_DIR / "val.csv"
            test_path = PROCESSED_DATA_DIR / "test.csv"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Saved splits to {PROCESSED_DATA_DIR}")
        except Exception as e:
            logger.error(f"Error saving splits: {str(e)}")
            raise
    
    def load_splits(self):
        """Load existing data splits"""
        try:
            train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
            val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
            test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
            
            logger.info("Loaded existing data splits")
            return train_df, val_df, test_df
        except FileNotFoundError:
            logger.warning("Split files not found. Creating new splits...")
            return None, None, None
