import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import os

def load_datasets():
    """Load all processed datasets"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/processed")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_df, val_df, test_df

def plot_label_distribution(df, title, save_path):
    """Plot distribution of labels"""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='label', order=['TRUE', 'FALSE', 'PARTIALLY TRUE'])
    plt.title(f"{title}\nTotal samples: {len(df)}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_source_distribution(df, title, save_path):
    """Plot distribution of sources"""
    plt.figure(figsize=(12, 6))
    source_counts = df['source'].value_counts()
    sns.barplot(x=source_counts.index, y=source_counts.values)
    plt.title(f"{title}\nTotal samples: {len(df)}")
    plt.xticks(rotation=45)
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_claim_length_distribution(df, title, save_path):
    """Plot distribution of claim lengths"""
    df['claim_length'] = df['claim'].str.len()
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='claim_length', bins=50)
    plt.title(f"{title}\nMean length: {df['claim_length'].mean():.1f} characters")
    plt.xlabel('Claim Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_wordcloud(df, title, save_path):
    """Create word cloud from claims"""
    text = ' '.join(df['claim'].astype(str))
    wordcloud = WordCloud(width=1200, height=800, 
                         background_color='white',
                         max_words=200,
                         collocations=False).generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_label_by_source(df, title, save_path):
    """Plot label distribution by source"""
    plt.figure(figsize=(14, 7))
    df_pivot = pd.crosstab(df['source'], df['label'], normalize='index') * 100
    df_pivot.plot(kind='bar', stacked=True)
    plt.title(f"{title}\nPercentage distribution")
    plt.xlabel('Source')
    plt.ylabel('Percentage')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    train_df, val_df, test_df = load_datasets()
    
    # Plot label distributions
    plot_label_distribution(train_df, "Label Distribution in Training Set", 
                          os.path.join(output_dir, "train_label_dist.png"))
    plot_label_distribution(val_df, "Label Distribution in Validation Set", 
                          os.path.join(output_dir, "val_label_dist.png"))
    plot_label_distribution(test_df, "Label Distribution in Test Set", 
                          os.path.join(output_dir, "test_label_dist.png"))
    
    # Plot source distributions
    plot_source_distribution(train_df, "Source Distribution in Training Set", 
                           os.path.join(output_dir, "train_source_dist.png"))
    plot_source_distribution(val_df, "Source Distribution in Validation Set", 
                           os.path.join(output_dir, "val_source_dist.png"))
    plot_source_distribution(test_df, "Source Distribution in Test Set", 
                           os.path.join(output_dir, "test_source_dist.png"))
    
    # Plot claim length distributions
    plot_claim_length_distribution(train_df, "Claim Length Distribution in Training Set", 
                                 os.path.join(output_dir, "train_claim_length_dist.png"))
    
    # Create word clouds
    create_wordcloud(train_df, "Most Common Words in Claims (Training Set)", 
                    os.path.join(output_dir, "train_wordcloud.png"))
    
    # Plot label distribution by source
    plot_label_by_source(train_df, "Label Distribution by Source (Training Set)", 
                        os.path.join(output_dir, "label_by_source.png"))
    
    # Create summary statistics
    summary = {
        'dataset_sizes': {
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df)
        },
        'label_distribution': {
            'train': train_df['label'].value_counts().to_dict(),
            'validation': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        },
        'source_distribution': {
            'train': train_df['source'].value_counts().to_dict(),
            'validation': val_df['source'].value_counts().to_dict(),
            'test': test_df['source'].value_counts().to_dict()
        },
        'avg_claim_length': {
            'train': train_df['claim'].str.len().mean(),
            'validation': val_df['claim'].str.len().mean(),
            'test': test_df['claim'].str.len().mean()
        }
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, "data_summary.txt"), 'w') as f:
        f.write("Dataset Summary Statistics\n")
        f.write("========================\n\n")
        
        f.write("Dataset Sizes:\n")
        for dataset, size in summary['dataset_sizes'].items():
            f.write(f"{dataset}: {size:,} examples\n")
        f.write("\n")
        
        f.write("Label Distribution:\n")
        for dataset, dist in summary['label_distribution'].items():
            f.write(f"\n{dataset}:\n")
            for label, count in dist.items():
                f.write(f"  {label}: {count:,}\n")
        f.write("\n")
        
        f.write("Source Distribution:\n")
        for dataset, dist in summary['source_distribution'].items():
            f.write(f"\n{dataset}:\n")
            for source, count in dist.items():
                f.write(f"  {source}: {count:,}\n")
        f.write("\n")
        
        f.write("Average Claim Length:\n")
        for dataset, length in summary['avg_claim_length'].items():
            f.write(f"{dataset}: {length:.2f} characters\n")

if __name__ == "__main__":
    main() 