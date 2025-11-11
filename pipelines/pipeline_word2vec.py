"""
Pipeline for Word2Vec embeddings + Random Forest
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
from src.embeddings.word2vec_embeddings import Word2VecEmbeddings
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.visualization.plot_utils import Visualizer
from src.utils.config import *
from src.utils.logger_utils import setup_logger

logger = setup_logger('pipeline_word2vec', log_dir=RESULTS_DIR)

def run_word2vec_pipeline():
    """Run complete pipeline with Word2Vec embeddings"""
    
    logger.info("="*80)
    logger.info("STARTING WORD2VEC + RANDOM FOREST PIPELINE")
    logger.info("="*80)
    
    # 1. Load Data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING")
    logger.info("="*80)
    
    data_loader = DataLoader()
    df = data_loader.load_data()
    data_loader.validate_data()
    
    # Check if splits exist
    train_df, val_df, test_df = data_loader.load_splits()
    if train_df is None:
        train_df, val_df, test_df = data_loader.split_data()
    
    # 2. Preprocess Data
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("="*80)
    
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)
    
    # Get tokens for training embeddings
    train_tokens = preprocessor.get_tokens_list(train_df)
    val_tokens = preprocessor.get_tokens_list(val_df)
    test_tokens = preprocessor.get_tokens_list(test_df)
    
    # 3. Generate Word2Vec Embeddings
    logger.info("\n" + "="*80)
    logger.info("STEP 3: GENERATING WORD2VEC EMBEDDINGS")
    logger.info("="*80)
    
    w2v_embedder = Word2VecEmbeddings()
    w2v_embedder.train(train_tokens)
    w2v_embedder.save_model('word2vec_model.pkl')
    
    # Transform to embeddings
    X_train = w2v_embedder.transform(train_tokens)
    X_val = w2v_embedder.transform(val_tokens)
    X_test = w2v_embedder.transform(test_tokens)
    
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
    trainer.save_model('random_forest_model.pkl', 'word2vec')
    
    # 5. Evaluate Model
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("="*80)
    
    evaluator = ModelEvaluator()
    
    train_metrics = evaluator.evaluate(model, X_train, y_train, 'train')
    val_metrics = evaluator.evaluate(model, X_val, y_val, 'validation')
    test_metrics = evaluator.evaluate(model, X_test, y_test, 'test')
    
    evaluator.save_metrics('word2vec')
    
    # 6. Visualizations
    logger.info("\n" + "="*80)
    logger.info("STEP 6: GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    visualizer = Visualizer()
    
    # Plot confusion matrix
    cm = np.array(test_metrics['confusion_matrix'])
    visualizer.plot_confusion_matrix(cm, 
                                    title='Word2Vec + RF Confusion Matrix',
                                    filename='word2vec_confusion_matrix.png')
    
    # Plot feature importance
    visualizer.plot_feature_importance(model, 
                                      filename='word2vec_feature_importance.png')
    
    logger.info("\n" + "="*80)
    logger.info("WORD2VEC PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return {
        'model': model,
        'metrics': evaluator.metrics,
        'embedder': w2v_embedder
    }

if __name__ == "__main__":
    results = run_word2vec_pipeline()
