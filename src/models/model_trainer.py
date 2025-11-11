"""
Model training utilities
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
from pathlib import Path
from src.utils.config import RF_PARAM_GRID, CV_FOLDS, MODELS_DIR
from src.utils.logger_utils import setup_logger

logger = setup_logger('model_trainer')

class ModelTrainer:
    """Class to train and tune Random Forest models"""
    
    def __init__(self, param_grid=None, cv_folds=CV_FOLDS, n_jobs=-1):
        """
        Initialize ModelTrainer
        
        Args:
            param_grid: Hyperparameter grid for tuning
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        self.param_grid = param_grid if param_grid else RF_PARAM_GRID
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.model = None
        self.best_params = None
        self.cv_results = None
        
    def train_without_tuning(self, X_train, y_train, random_state=42):
        """
        Train Random Forest without hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random state
        
        Returns:
            model: Trained model
        """
        logger.info("Training Random Forest without hyperparameter tuning...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=self.n_jobs
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Training completed")
        
        return self.model
    
    def train_with_tuning(self, X_train, y_train, X_val, y_val):
        """
        Train Random Forest with hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            best_model: Best model from grid search
        """
        logger.info("Training Random Forest with hyperparameter tuning...")
        logger.info(f"Parameter grid: {self.param_grid}")
        
        # Create base model
        rf = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=self.param_grid,
            cv=self.cv_folds,
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=2
        )
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        val_score = self.model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return self.model
    
    def save_model(self, filename, embedding_type):
        """
        Save trained model
        
        Args:
            filename: Filename for the model
            embedding_type: Type of embedding used
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = MODELS_DIR / f"{embedding_type}_{filename}"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save best params if available
        if self.best_params:
            params_file = MODELS_DIR / f"{embedding_type}_best_params.pkl"
            joblib.dump(self.best_params, params_file)
            logger.info(f"Best parameters saved to {params_file}")
    
    def load_model(self, filename, embedding_type):
        """Load trained model"""
        filepath = MODELS_DIR / f"{embedding_type}_{filename}"
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model
