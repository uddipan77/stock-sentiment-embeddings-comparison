"""
Main script to run all pipelines and compare results
"""
import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.pipeline_word2vec import run_word2vec_pipeline
from pipelines.pipeline_glove import run_glove_pipeline
from pipelines.pipeline_fasttext import run_fasttext_pipeline
from pipelines.pipeline_sentence_transformer import run_sentence_transformer_pipeline
from src.models.model_evaluator import ModelEvaluator
from src.visualization.plot_utils import Visualizer
from src.utils.config import *
from src.utils.logger_utils import setup_logger

logger = setup_logger('main', log_dir=RESULTS_DIR)

def run_all_pipelines():
    """Run all embedding pipelines"""
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING ALL SENTIMENT ANALYSIS PIPELINES")
    logger.info("="*80)
    
    results = {}
    
    # 1. Word2Vec Pipeline
    try:
        logger.info("\n\n### Running Word2Vec Pipeline ###\n")
        results['word2vec'] = run_word2vec_pipeline()
    except Exception as e:
        logger.error(f"Word2Vec pipeline failed: {str(e)}")
        results['word2vec'] = None
    
    # 2. GloVe Pipeline
    try:
        logger.info("\n\n### Running GloVe Pipeline ###\n")
        results['glove'] = run_glove_pipeline()
    except Exception as e:
        logger.error(f"GloVe pipeline failed: {str(e)}")
        results['glove'] = None
    
    # 3. FastText Pipeline
    try:
        logger.info("\n\n### Running FastText Pipeline ###\n")
        results['fasttext'] = run_fasttext_pipeline()
    except Exception as e:
        logger.error(f"FastText pipeline failed: {str(e)}")
        results['fasttext'] = None
    
    # 4. Sentence Transformer Pipeline
    try:
        logger.info("\n\n### Running Sentence Transformer Pipeline ###\n")
        results['sentence_transformer'] = run_sentence_transformer_pipeline()
    except Exception as e:
        logger.error(f"Sentence Transformer pipeline failed: {str(e)}")
        results['sentence_transformer'] = None
    
    return results

def compare_all_models(results):
    """Compare results from all models"""
    
    logger.info("\n" + "="*80)
    logger.info("COMPARING ALL MODELS")
    logger.info("="*80)
    
    # Collect metrics
    metrics_list = []
    model_names = []
    
    for name, result in results.items():
        if result is not None and 'metrics' in result:
            metrics_list.append(result['metrics'])
            model_names.append(name.replace('_', ' ').title())
    
    if len(metrics_list) == 0:
        logger.error("No successful pipelines to compare")
        return
    
    # Create comparison
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(metrics_list, model_names)
    
    # Save comparison
    comparison_path = METRICS_DIR / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Create comparison visualization
    visualizer = Visualizer()
    visualizer.plot_metrics_comparison(comparison_df, 'final_model_comparison.png')
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL MODEL COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    logger.info(f"\n🏆 BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
    
    return comparison_df

def main():
    """Main execution function"""
    
    logger.info("="*80)
    logger.info("STOCK SENTIMENT ANALYSIS - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Results Directory: {RESULTS_DIR}")
    
    # Run all pipelines
    results = run_all_pipelines()
    
    # Compare models
    if any(r is not None for r in results.values()):
        comparison_df = compare_all_models(results)
    else:
        logger.error("All pipelines failed. Cannot perform comparison.")
    
    logger.info("\n" + "="*80)
    logger.info("ALL PIPELINES COMPLETED")
    logger.info("="*80)
    logger.info(f"\nResults saved in: {RESULTS_DIR}")
    logger.info(f"- Models: {MODELS_DIR}")
    logger.info(f"- Metrics: {METRICS_DIR}")
    logger.info(f"- Plots: {PLOTS_DIR}")
    logger.info(f"- Embeddings: {EMBEDDINGS_DIR}")

if __name__ == "__main__":
    main()
