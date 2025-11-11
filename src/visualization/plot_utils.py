"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from src.utils.config import PLOTS_DIR, LABEL_MAPPING
from src.utils.logger_utils import setup_logger

logger = setup_logger('plot_utils')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

class Visualizer:
    """Class for creating visualizations"""
    
    def __init__(self, save_dir=PLOTS_DIR):
        """
        Initialize Visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_label_distribution(self, y, title='Label Distribution', filename='label_distribution.png'):
        """
        Plot distribution of labels
        
        Args:
            y: Label array
            title: Plot title
            filename: Filename to save plot
        """
        plt.figure(figsize=(8, 6))
        
        unique_labels, counts = np.unique(y, return_counts=True)
        label_names = [LABEL_MAPPING.get(label, str(label)) for label in unique_labels]
        
        colors = ['red', 'gray', 'green']
        plt.bar(label_names, counts, color=colors[:len(label_names)])
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=0)
        
        # Add count labels on bars
        for i, (label, count) in enumerate(zip(label_names, counts)):
            plt.text(i, count + max(counts)*0.01, str(count), 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Label distribution plot saved to {filepath}")
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix', filename='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            title: Plot title
            filename: Filename to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Get label names
        n_classes = cm.shape[0]
        if n_classes == 3:
            labels = ['Negative', 'Neutral', 'Positive']
        else:
            labels = [str(i) for i in range(n_classes)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix plot saved to {filepath}")
    
    def plot_metrics_comparison(self, comparison_df, filename='metrics_comparison.png'):
        """
        Plot comparison of metrics across models
        
        Args:
            comparison_df: DataFrame with model comparison
            filename: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=colors[idx], alpha=0.7)
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Metrics comparison plot saved to {filepath}")
    
    def plot_feature_importance(self, model, top_n=20, filename='feature_importance.png'):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_
            top_n: Number of top features to display
            filename: Filename to save plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='skyblue')
        plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14)
        plt.tight_layout()
        
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {filepath}")
