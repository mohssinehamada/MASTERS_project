import pandas as pd
import json
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
import re

def load_fever_dataset(data_dir="data/raw"):
    """
    Load the FEVER dataset from the raw data directory
    """
    try:
        fever_path = os.path.join(data_dir, "fever")
        if not os.path.exists(fever_path):
            print(f"FEVER dataset not found at {fever_path}. Please download it first.")
            return []
        
        # Load FEVER data
        train_file = os.path.join(fever_path, "train.jsonl")
        data = []
        
        with open(train_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # Convert FEVER labels to our format
                label_map = {
                    "SUPPORTS": "TRUE",
                    "REFUTES": "FALSE",
                    "NOT ENOUGH INFO": "UNVERIFIABLE"
                }
                
                if item['label'] in label_map:
                    data.append({
                        'claim': item['claim'],
                        'evidence': ' '.join([ev[2] for ev in item.get('evidence', [])]),
                        'label': label_map[item['label']],
                        'explanation': '',  # FEVER doesn't provide explanations
                        'source': 'FEVER'
                    })
        
        return data
        
    except Exception as e:
        print(f"Error loading FEVER dataset: {e}")
        return []

def load_liar_dataset(data_dir="data/raw"):
    """
    Load the LIAR dataset from the raw data directory
    """
    try:
        liar_path = os.path.join(data_dir, "liar")
        if not os.path.exists(liar_path):
            print(f"LIAR dataset not found at {liar_path}. Please download it first.")
            return []
        
        # Load LIAR data (TSV format)
        train_file = os.path.join(liar_path, "train.tsv")
        data = []
        
        with open(train_file, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) >= 14:  # LIAR has 14 columns
                    # Map LIAR labels to our format
                    label_map = {
                        "pants-fire": "FALSE",
                        "false": "FALSE",
                        "barely-true": "PARTIALLY TRUE",
                        "half-true": "PARTIALLY TRUE",
                        "mostly-true": "PARTIALLY TRUE",
                        "true": "TRUE"
                    }
                    
                    if cols[1] in label_map:
                        data.append({
                            'claim': cols[2],
                            'evidence': cols[3],  # Statement context
                            'label': label_map[cols[1]],
                            'explanation': cols[4],  # Justification
                            'source': 'LIAR'
                        })
        
        return data
        
    except Exception as e:
        print(f"Error loading LIAR dataset: {e}")
        return []

def load_politifact_dataset(data_dir="data/raw"):
    """
    Load a processed PolitiFact dataset if available
    """
    try:
        politifact_path = os.path.join(data_dir, "politifact", "politifact.csv")
        if not os.path.exists(politifact_path):
            print(f"PolitiFact dataset not found at {politifact_path}. Please download it first.")
            return []
        
        # Load PolitiFact data
        df = pd.read_csv(politifact_path)
        data = []
        
        for _, row in df.iterrows():
            # Map PolitiFact ratings to our format
            label_map = {
                "true": "TRUE",
                "mostly true": "PARTIALLY TRUE",
                "half true": "PARTIALLY TRUE",
                "barely true": "PARTIALLY TRUE",
                "false": "FALSE",
                "pants on fire": "FALSE"
            }
            
            fact = str(row.get('fact', '')).lower()
            if fact in label_map:
                data.append({
                    'claim': str(row.get('sources_quote', '')).strip(),
                    'evidence': str(row.get('curator_complete_article', '')),
                    'label': label_map[fact],
                    'explanation': str(row.get('curators_article_title', '')),
                    'source': 'PolitiFact'
                })
        
        return data
        
    except Exception as e:
        print(f"Error loading PolitiFact dataset: {e}")
        return []

def load_and_prepare_datasets(data_dir="data/raw", processed_dir="data/processed"):
    """Load and prepare datasets for fine-tuning"""
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load datasets
    datasets = [
        load_fever_dataset(data_dir),
        load_liar_dataset(data_dir),
        load_politifact_dataset(data_dir),
    ]
    
    # Combine and standardize format
    combined_data = []
    for dataset in datasets:
        combined_data.extend(dataset)
    
    if not combined_data:
        raise ValueError("No data was loaded. Please check your datasets.")
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(combined_data)
    
    # Balance the dataset
    # Get counts for each label
    label_counts = df['label'].value_counts()
    min_count = min(label_counts.values)
    
    # Sample each label to the minimum count
    balanced_dfs = []
    for label in label_counts.index:
        label_df = df[df['label'] == label]
        balanced_dfs.append(label_df.sample(min_count, replace=False, random_state=42))
    
    # Combine balanced datasets
    balanced_df = pd.concat(balanced_dfs)
    
    # Create train/val/test split
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    
    # Save processed datasets
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print(f"Created dataset with {len(train_dataset)} training, {len(val_dataset)} validation, "
          f"and {len(test_dataset)} test examples")
    
    return dataset_dict

def create_prompt_response_pairs(examples):
    """
    Convert datasets into instruction tuning format
    
    Args:
        examples: Dataset examples with claim, evidence, label, explanation fields
        
    Returns:
        List of dictionaries with 'instruction' and 'response' fields
    """
    formatted_data = []
    
    for example in examples:
        # Clean up the evidence
        evidence = example['evidence'] if example['evidence'] else "No additional context provided."
        
        # Create instruction following format
        prompt = f"""Analyze the following claim and determine if it is true or false:
        
Claim: {example['claim']}

Context: {evidence}

Based on the provided context, perform a factual analysis of the claim.
Provide a verdict (TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE) and explain your reasoning.
"""
        
        # Create expected response
        explanation = example['explanation'] if example['explanation'] else "Based on the available evidence."
        
        response = f"""Verdict: {example['label']}

Reasoning: {explanation}
"""
        
        formatted_data.append({
            "instruction": prompt,
            "response": response,
        })
    
    return formatted_data

def download_datasets():
    """
    Function to download datasets from public sources if not already present
    """
    # This would typically connect to dataset repositories and download files
    # For now, just create the directory structure and a note
    
    os.makedirs("data/raw/fever", exist_ok=True)
    os.makedirs("data/raw/liar", exist_ok=True)
    os.makedirs("data/raw/politifact", exist_ok=True)
    
    readme_path = "data/raw/README.md"
    
    with open(readme_path, 'w') as f:
        f.write("""# Dataset Downloads

Please download the following datasets and place them in the appropriate directories:

1. FEVER dataset: https://fever.ai/dataset/fever.html
   - Place in data/raw/fever/

2. LIAR dataset: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
   - Extract and place in data/raw/liar/

3. PolitiFact dataset: You'll need to scrape or find a pre-processed version
   - Place in data/raw/politifact/

Once downloaded, run:
```
python -m utils.preprocessing
```
To prepare the data for training.
""")
    
    print(f"Created dataset directories and instructions in {readme_path}")
    
if __name__ == "__main__":
    # If run directly, try to process the data
    download_datasets()
    try:
        dataset = load_and_prepare_datasets()
        print("Dataset preparation complete.")
    except Exception as e:
        print(f"Error preparing datasets: {e}")
        print("Please make sure you've downloaded the required datasets first.") 