"""
Exploratory Data Analysis script
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.data_preprocessing import TextPreprocessor
from src.utils.config import *
from src.utils.logger_utils import setup_logger

logger = setup_logger('eda', log_dir=RESULTS_DIR)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class EDAAnalyzer:
    """Class for Exploratory Data Analysis"""
    
    def __init__(self, df):
        """
        Initialize EDA Analyzer
        
        Args:
            df: Input dataframe
        """
        self.df = df
        self.save_dir = PLOTS_DIR / 'eda'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def basic_info(self):
        """Display basic information about the dataset"""
        logger.info("\n" + "="*80)
        logger.info("BASIC DATASET INFORMATION")
        logger.info("="*80)
        
        logger.info(f"\nDataset Shape: {self.df.shape}")
        logger.info(f"Number of samples: {len(self.df)}")
        logger.info(f"Number of features: {len(self.df.columns)}")
        
        logger.info(f"\nColumns: {self.df.columns.tolist()}")
        logger.info(f"\nData Types:\n{self.df.dtypes}")
        
        logger.info(f"\nMissing Values:\n{self.df.isnull().sum()}")
        logger.info(f"\nDuplicate Rows: {self.df.duplicated().sum()}")
        
        logger.info(f"\nFirst few rows:\n{self.df.head()}")
        
    def analyze_labels(self):
        """Analyze label distribution"""
        logger.info("\n" + "="*80)
        logger.info("LABEL DISTRIBUTION ANALYSIS")
        logger.info("="*80)
        
        label_counts = self.df['Label'].value_counts().sort_index()
        logger.info(f"\nLabel Counts:\n{label_counts}")
        
        label_percentages = (label_counts / len(self.df) * 100).round(2)
        logger.info(f"\nLabel Percentages:\n{label_percentages}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        colors = ['red', 'gray', 'green']
        label_names = [LABEL_MAPPING.get(label, str(label)) for label in label_counts.index]
        axes[0].bar(label_names, label_counts.values, color=colors)
        axes[0].set_xlabel('Sentiment', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Label Distribution (Counts)', fontsize=14, fontweight='bold')
        
        for i, (name, count) in enumerate(zip(label_names, label_counts.values)):
            axes[0].text(i, count + max(label_counts.values)*0.01, 
                        str(count), ha='center', va='bottom', fontsize=11)
        
        # Pie chart
        axes[1].pie(label_counts.values, labels=label_names, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Label distribution plot saved to {self.save_dir / 'label_distribution.png'}")
    
    def analyze_stock_prices(self):
        """Analyze stock price features"""
        logger.info("\n" + "="*80)
        logger.info("STOCK PRICE ANALYSIS")
        logger.info("="*80)
        
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if all(col in self.df.columns for col in price_cols):
            logger.info(f"\nStock Price Statistics:\n{self.df[price_cols].describe()}")
            
            # Time series plot
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Price trends
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values('Date')
            
            axes[0].plot(self.df['Date'], self.df['Open'], label='Open', alpha=0.7)
            axes[0].plot(self.df['Date'], self.df['High'], label='High', alpha=0.7)
            axes[0].plot(self.df['Date'], self.df['Low'], label='Low', alpha=0.7)
            axes[0].plot(self.df['Date'], self.df['Close'], label='Close', alpha=0.7)
            axes[0].set_xlabel('Date', fontsize=12)
            axes[0].set_ylabel('Price ($)', fontsize=12)
            axes[0].set_title('Stock Price Trends', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Volume
            axes[1].bar(self.df['Date'], self.df['Volume'], color='steelblue', alpha=0.6)
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].set_ylabel('Volume', fontsize=12)
            axes[1].set_title('Trading Volume Over Time', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'stock_prices.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Stock price plot saved to {self.save_dir / 'stock_prices.png'}")
    
    def analyze_text_length(self):
        """Analyze news text length"""
        logger.info("\n" + "="*80)
        logger.info("TEXT LENGTH ANALYSIS")
        logger.info("="*80)
        
        self.df['text_length'] = self.df['News'].apply(lambda x: len(str(x)))
        self.df['word_count'] = self.df['News'].apply(lambda x: len(str(x).split()))
        
        logger.info(f"\nText Length Statistics:\n{self.df[['text_length', 'word_count']].describe()}")
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Text length distribution
        axes[0, 0].hist(self.df['text_length'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Character Count', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Text Length Distribution', fontsize=12, fontweight='bold')
        
        # Word count distribution
        axes[0, 1].hist(self.df['word_count'], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Word Count', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
        
        # Text length by sentiment
        label_names = [LABEL_MAPPING.get(label, str(label)) for label in sorted(self.df['Label'].unique())]
        for label in sorted(self.df['Label'].unique()):
            label_name = LABEL_MAPPING.get(label, str(label))
            subset = self.df[self.df['Label'] == label]['text_length']
            axes[1, 0].hist(subset, bins=30, alpha=0.5, label=label_name)
        axes[1, 0].set_xlabel('Character Count', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Text Length by Sentiment', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        
        # Box plot
        data_by_label = [self.df[self.df['Label'] == label]['word_count'].values 
                        for label in sorted(self.df['Label'].unique())]
        axes[1, 1].boxplot(data_by_label, labels=label_names)
        axes[1, 1].set_xlabel('Sentiment', fontsize=11)
        axes[1, 1].set_ylabel('Word Count', fontsize=11)
        axes[1, 1].set_title('Word Count Distribution by Sentiment', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Text length analysis plot saved to {self.save_dir / 'text_length_analysis.png'}")
    
    def analyze_word_frequencies(self):
        """Analyze word frequencies"""
        logger.info("\n" + "="*80)
        logger.info("WORD FREQUENCY ANALYSIS")
        logger.info("="*80)
        
        # Preprocess text
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        all_words = []
        label_words = {label: [] for label in self.df['Label'].unique()}
        
        for idx, row in self.df.iterrows():
            processed = preprocessor.preprocess_text(row['News'])
            words = processed.split()
            all_words.extend(words)
            label_words[row['Label']].extend(words)
        
        # Overall word frequency
        word_freq = Counter(all_words)
        top_20 = word_freq.most_common(20)
        
        logger.info(f"\nTop 20 Most Common Words:")
        for word, count in top_20:
            logger.info(f"{word}: {count}")
        
        # Plot word frequencies
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall word frequency bar chart
        words, counts = zip(*top_20)
        axes[0, 0].barh(range(len(words)), counts, color='steelblue')
        axes[0, 0].set_yticks(range(len(words)))
        axes[0, 0].set_yticklabels(words)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Top 20 Most Common Words', fontsize=12, fontweight='bold')
        
        # Word cloud for all text
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis').generate(' '.join(all_words))
        axes[0, 1].imshow(wordcloud, interpolation='bilinear')
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Word Cloud (All Text)', fontsize=12, fontweight='bold')
        
        # Word clouds by sentiment
        for idx, (label, words) in enumerate(sorted(label_words.items())):
            if idx >= 2:
                break
            label_name = LABEL_MAPPING.get(label, str(label))
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                 colormap='viridis').generate(' '.join(words))
            axes[1, idx].imshow(wordcloud, interpolation='bilinear')
            axes[1, idx].axis('off')
            axes[1, idx].set_title(f'Word Cloud ({label_name})', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'word_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word frequency analysis plot saved to {self.save_dir / 'word_frequency_analysis.png'}")
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        logger.info("\n" + "="*80)
        logger.info("CORRELATION ANALYSIS")
        logger.info("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            logger.info(f"\nCorrelation Matrix:\n{corr_matrix}")
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, linewidths=1)
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.save_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved to {self.save_dir / 'correlation_heatmap.png'}")
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE EDA")
        logger.info("="*80)
        
        self.basic_info()
        self.analyze_labels()
        self.analyze_stock_prices()
        self.analyze_text_length()
        self.analyze_word_frequencies()
        self.correlation_analysis()
        
        logger.info("\n" + "="*80)
        logger.info("EDA COMPLETED SUCCESSFULLY")
        logger.info("="*80)

def main():
    """Main function to run EDA"""
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Run EDA
    eda_analyzer = EDAAnalyzer(df)
    eda_analyzer.run_complete_eda()

if __name__ == "__main__":
    main()
