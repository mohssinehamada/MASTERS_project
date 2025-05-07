import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from tqdm import tqdm
from datasets import load_dataset

def extract_verdict(text):
    """
    Extract the verdict from model output text
    """
    # Look for the verdict pattern
    verdict_pattern = r"Verdict:\s*(TRUE|FALSE|PARTIALLY TRUE|UNVERIFIABLE)"
    match = re.search(verdict_pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    # Fallback patterns
    if "true" in text.lower() and not "false" in text.lower():
        return "TRUE"
    elif "false" in text.lower():
        return "FALSE"
    elif "partially" in text.lower() or "partly" in text.lower():
        return "PARTIALLY TRUE"
    else:
        return "UNVERIFIABLE"

def evaluate_model(model, tokenizer, test_data, output_dir="data/evaluation"):
    """
    Evaluate model performance on test data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Make sure model is in evaluation mode
    model.eval()
    
    results = []
    ground_truth = []
    predictions = []
    
    # Evaluate each example
    for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
        claim = example["claim"]
        evidence = example.get("evidence", "")
        true_label = example["label"]
        
        # Create prompt
        prompt = f"""Analyze the following claim and determine if it is true or false:
        
Claim: {claim}

Context: {evidence}

Based on the provided context, perform a factual analysis of the claim.
Provide a verdict (TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE) and explain your reasoning.
"""
        
        # Generate response with truncation to match training configuration
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=512, 
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predicted verdict
        predicted_label = extract_verdict(response)
        
        # Store results
        results.append({
            "claim": claim,
            "evidence": evidence,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "full_response": response,
            "correct": true_label == predicted_label
        })
        
        ground_truth.append(true_label)
        predictions.append(predicted_label)
    
    # Save detailed results
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted')
    
    # Get class-specific metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average=None)
    
    # Create confusion matrix
    labels = sorted(list(set(ground_truth + predictions)))
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Save summary metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_metrics": {
            label: {
                "precision": float(class_precision[i]),
                "recall": float(class_recall[i]),
                "f1": float(class_f1[i])
            } for i, label in enumerate(labels)
        }
    }
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClass-specific metrics:")
    for i, label in enumerate(labels):
        print(f"{label}: Precision={class_precision[i]:.4f}, Recall={class_recall[i]:.4f}, F1={class_f1[i]:.4f}")
    
    return metrics

def load_test_data(test_file="data/processed/test.csv"):
    """Load test data from processed file"""
    if os.path.exists(test_file):
        return pd.read_csv(test_file).to_dict('records')
    else:
        # Try to load from HuggingFace datasets
        try:
            dataset = load_dataset('csv', data_files={'test': test_file})
            return dataset['test']
        except:
            raise FileNotFoundError(f"Test data not found at {test_file}")

def main():
    """
    Main evaluation function
    """
    model_path = "model/checkpoints/final"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    
    # Evaluate model
    print("Starting evaluation...")
    metrics = evaluate_model(model, tokenizer, test_data)
    
    print(f"\nEvaluation complete. Results saved to data/evaluation/")

if __name__ == "__main__":
    main() 