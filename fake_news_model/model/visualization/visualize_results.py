import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import json
import os

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('fake_news_model/model/checkpoints/metrics/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_scores, labels):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Convert labels to one-hot encoding
    y_true_bin = pd.get_dummies(y_true)
    
    # Ensure all labels are present in y_true_bin
    for label in labels:
        if label not in y_true_bin.columns:
            y_true_bin[label] = 0
    
    # Reorder columns to match labels
    y_true_bin = y_true_bin[labels]
    
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[label], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('fake_news_model/model/checkpoints/metrics/roc_curves.png')
    plt.close()

def plot_metrics(metrics):
    """Plot various performance metrics"""
    # Create bar plot for accuracy, precision, recall, F1-score
    metrics_df = pd.DataFrame(metrics).T
    
    # Filter out unwanted rows and columns
    metrics_df = metrics_df[metrics_df.index.isin(['TRUE', 'FALSE', 'PARTIALLY TRUE', 'UNVERIFIABLE'])]
    metrics_df = metrics_df[['precision', 'recall', 'f1-score']]
    
    # Replace NaN with 0
    metrics_df = metrics_df.fillna(0)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar')
    plt.title('Performance Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fake_news_model/model/checkpoints/metrics/performance_metrics.png')
    plt.close()

def plot_confidence_distribution(confidences):
    """Plot distribution of confidence scores"""
    plt.figure(figsize=(10, 6))
    sns.histplot(confidences, bins=20)
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.savefig('fake_news_model/model/checkpoints/metrics/confidence_distribution.png')
    plt.close()

def main():
    # Create metrics directory if it doesn't exist
    os.makedirs('fake_news_model/model/checkpoints/metrics', exist_ok=True)
    
    # Load results from test_model.py output
    try:
        with open('fake_news_model/model/checkpoints/metrics/test_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("No test results found. Please run test_model.py first.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in test results file.")
        return
    
    # Extract data for visualization
    try:
        # Handle the format where we have true_labels, predictions, and prediction_scores as top-level keys
        if isinstance(results, dict) and 'predictions' in results:
            # Load true labels from test claims
            try:
                with open('fake_news_model/data/test_claims.json', 'r') as f:
                    test_claims = json.load(f)
                    y_true = [claim['true_label'] for claim in test_claims]
            except Exception as e:
                print(f"Warning: Could not load true labels from test claims: {e}")
                y_true = ['UNVERIFIABLE'] * len(results['predictions'])  # Fallback
            
            y_pred = results['predictions']
            y_scores = np.array(results['prediction_scores'])
            
            # Ensure y_scores is in the right format (n_samples x n_classes)
            if len(y_scores.shape) == 2 and y_scores.shape[0] == len(y_pred):
                confidences = np.max(y_scores, axis=1)
            else:
                print("Warning: Prediction scores are not in the expected format. Using default confidence values.")
                confidences = np.ones(len(y_pred))  # Default confidence of 1.0
        else:
            print("Error: Unexpected results format. Expected a dict with 'predictions' key.")
            return
        
        print("Data extracted successfully:")
        print(f"Number of samples: {len(y_true)}")
        print(f"Unique true labels: {set(y_true)}")
        print(f"Unique predicted labels: {set(y_pred)}")
        
        # Define labels (use a fixed set to ensure consistent visualization)
        labels = ['TRUE', 'FALSE', 'PARTIALLY TRUE', 'UNVERIFIABLE']
        print(f"Using labels: {labels}")
        
        # Generate visualizations
        plot_confusion_matrix(y_true, y_pred, labels)
        plot_roc_curve(y_true, y_scores, labels)
        
        # Calculate and plot metrics with zero_division=0
        metrics = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        plot_metrics(metrics)
        
        # Plot confidence distribution
        plot_confidence_distribution(confidences)
        
        print("Visualizations have been saved to fake_news_model/model/checkpoints/metrics/")
        
        # Print detailed metrics
        print("\nDetailed Metrics:")
        print("----------------")
        for label in labels:
            print(f"\n{label}:")
            if label in metrics:
                print(f"  Precision: {metrics[label]['precision']:.2f}")
                print(f"  Recall: {metrics[label]['recall']:.2f}")
                print(f"  F1-score: {metrics[label]['f1-score']:.2f}")
            else:
                print("  No predictions for this class")
        
        print("\nOverall Accuracy:", metrics['accuracy'])
        
    except KeyError as e:
        print(f"Error: Missing required field in results: {e}")
        print("Available keys:", results.keys() if isinstance(results, dict) else "N/A")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 