"""
Pipeline for Sentence Transformer embeddings + Random Forest
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.data_preprocessing import TextPreprocessor
from src.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.visualization.plot_utils import Visualizer
from src.utils.config import *
from src.utils.logger_utils import setup_logger

logger = setup_logger('pipeline_sentence_transformer', log_dir=RESULTS_DIR)

def run_sentence_transformer_pipeline():
    """Run complete pipeline with Sentence Transformer embeddings"""
    
    logger.info("="*80)
    logger.info("STARTING SENTENCE TRANSFORMER + RANDOM FOREST PIPELINE")
    logger.info("="*80)
    
    # 1. Load Data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING")
    logger.info("="*80)
    
    data_loader = DataLoader()
    
    # Load existing splits
    train_df, val_df, test_df = data_loader.load_splits()
    if train_df is None:
        df = data_loader.load_data()
        data_loader.validate_data()
        train_df, val_df, test_df = data_loader.split_data()
    
    # 2. Preprocess Data (lighter preprocessing for transformers)
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("="*80)
    
    # For sentence transformers, use minimal preprocessing
    preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=False)
    
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)
    
    # Use processed text directly (not tokenized)
    train_texts = train_df['processed_text'].tolist()
    val_texts = val_df['processed_text'].tolist()
    test_texts = test_df['processed_text'].tolist()
    
    # 3. Generate Sentence Transformer Embeddings
    logger.info("\n" + "="*80)
    logger.info("STEP 3: GENERATING SENTENCE TRANSFORMER EMBEDDINGS")
    logger.info("="*80)
    
    st_embedder = SentenceTransformerEmbeddings()
    st_embedder.load_model()
    
    # Transform to embeddings
    X_train = st_embedder.transform(train_texts)
    X_val = st_embedder.transform(val_texts)
    X_test = st_embedder.transform(test_texts)
    
    y_train = train_df['Label'].values
    y_val = val_df['Label'].values
    y_test = test_df['Label'].values
    
    logger.info(f"Training embeddings shape: {X_train.shape}")
    logger.info(f"Validation embeddings shape: {X_val.shape}")
    logger.info(f"Test embeddings shape: {X_test.shape}")
    
    # 4. Train Random Forest with Hyperparameter Tuning
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAINING RANDOM FOREST WITH HYPERPARAMETER TUNING")
    logger.info("="*80)
    
    trainer = ModelTrainer()
    model = trainer.train_with_tuning(X_train, y_train, X_val, y_val)
    trainer.save_model('random_forest_model.pkl', 'sentence_transformer')
    
    # 5. Evaluate Model
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("="*80)
    
    evaluator = ModelEvaluator()
    
    train_metrics = evaluator.evaluate(model, X_train, y_train, 'train')
    val_metrics = evaluator.evaluate(model, X_val, y_val, 'validation')
    test_metrics = evaluator.evaluate(model, X_test, y_test, 'test')
    
    evaluator.save_metrics('sentence_transformer')
    
    # 6. Visualizations
    logger.info("\n" + "="*80)
    logger.info("STEP 6: GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    visualizer = Visualizer()
    
    # Plot confusion matrix
    cm = np.array(test_metrics['confusion_matrix'])
    visualizer.plot_confusion_matrix(cm, 
                                    title='Sentence Transformer + RF Confusion Matrix',
                                    filename='sentence_transformer_confusion_matrix.png')
    
    # Plot feature importance
    visualizer.plot_feature_importance(model, 
                                      filename='sentence_transformer_feature_importance.png')
    
    logger.info("\n" + "="*80)
    logger.info("SENTENCE TRANSFORMER PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return {
        'model': model,
        'metrics': evaluator.metrics,
        'embedder': st_embedder
    }

if __name__ == "__main__":
    results = run_sentence_transformer_pipeline()
