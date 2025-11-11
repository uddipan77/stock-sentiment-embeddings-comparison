"""
Model evaluation utilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import json
from pathlib import Path
from src.utils.config import METRICS_DIR, LABEL_MAPPING
from src.utils.logger_utils import setup_logger

logger = setup_logger('model_evaluator')

class ModelEvaluator:
    """Class to evaluate classification models"""
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        self.metrics = {}
        
    def evaluate(self, model, X, y, dataset_name='test'):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name of the dataset (train/val/test)
        
        Returns:
            metrics_dict: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set...")
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics_dict = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        per_class_precision = precision_score(y, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0)
        
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            label_name = LABEL_MAPPING.get(label, str(label))
            metrics_dict[f'precision_{label_name}'] = per_class_precision[i]
            metrics_dict[f'recall_{label_name}'] = per_class_recall[i]
            metrics_dict[f'f1_{label_name}'] = per_class_f1[i]
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics_dict['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y, y_pred, target_names=[LABEL_MAPPING.get(l, str(l)) for l in unique_labels])
        logger.info(f"\nClassification Report for {dataset_name} set:\n{report}")
        
        # Log main metrics
        logger.info(f"\n{dataset_name.upper()} SET METRICS:")
        logger.info(f"Accuracy: {metrics_dict['accuracy']:.4f}")
        logger.info(f"Precision (weighted): {metrics_dict['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {metrics_dict['recall_weighted']:.4f}")
        logger.info(f"F1 Score (weighted): {metrics_dict['f1_weighted']:.4f}")
        
        self.metrics[dataset_name] = metrics_dict
        return metrics_dict
    
    def save_metrics(self, embedding_type, filename='metrics.json'):
        """
        Save metrics to file
        
        Args:
            embedding_type: Type of embedding used
            filename: Filename for metrics
        """
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        filepath = METRICS_DIR / f"{embedding_type}_{filename}"
        
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for key, value in self.metrics.items():
            metrics_json[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.floating)):
                    metrics_json[key][k] = float(v)
                elif isinstance(v, np.ndarray):
                    metrics_json[key][k] = v.tolist()
                elif isinstance(v, list):
                    metrics_json[key][k] = v
                else:
                    metrics_json[key][k] = v
        
        with open(filepath, 'w') as f:
            json.dump(metrics_json, f, indent=4)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def compare_models(self, metrics_list, model_names):
        """
        Compare metrics across different models
        
        Args:
            metrics_list: List of metrics dictionaries
            model_names: List of model names
        
        Returns:
            comparison_df: DataFrame comparing models
        """
        comparison_data = []
        
        for metrics, name in zip(metrics_list, model_names):
            if 'test' in metrics:
                test_metrics = metrics['test']
                comparison_data.append({
                    'Model': name,
                    'Accuracy': test_metrics['accuracy'],
                    'Precision': test_metrics['precision_weighted'],
                    'Recall': test_metrics['recall_weighted'],
                    'F1-Score': test_metrics['f1_weighted']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df
