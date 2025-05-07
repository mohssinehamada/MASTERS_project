import json
import random
from collections import Counter
from typing import List, Dict, Any
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
import ijson  # For streaming JSON parsing
import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file using streaming parser."""
    data = []
    with open(file_path, 'rb') as f:
        # Parse the JSON array
        parser = ijson.items(f, 'item')
        for item in tqdm(parser, desc="Loading data"):
            data.append(item)
    return data

def save_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_class_distribution(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get the distribution of classes in the dataset."""
    labels = []
    for item in data:
        response = item.get('response', '')
        if 'Verdict:' in response:
            label = response.split('Verdict:')[1].split('\n')[0].strip()
            labels.append(label)
    return dict(Counter(labels))

def oversample_minority_classes(data: List[Dict[str, Any]], target_ratio: float = 0.3) -> List[Dict[str, Any]]:
    """Oversample minority classes to reach target ratio."""
    class_dist = get_class_distribution(data)
    majority_class = max(class_dist.items(), key=lambda x: x[1])[0]
    majority_count = class_dist[majority_class]
    
    balanced_data = data.copy()
    
    for label, count in class_dist.items():
        if label != majority_class:
            target_count = int(majority_count * target_ratio)
            if count < target_count:
                # Get samples of this class
                class_samples = [item for item in data if 'Verdict: ' + label in item['response']]
                # Calculate how many more samples we need
                n_samples_needed = target_count - count
                # Randomly sample with replacement
                new_samples = random.choices(class_samples, k=n_samples_needed)
                balanced_data.extend(new_samples)
    
    return balanced_data

def undersample_majority_class(data: List[Dict[str, Any]], target_ratio: float = 0.3) -> List[Dict[str, Any]]:
    """Undersample majority class to match target ratio."""
    class_dist = get_class_distribution(data)
    majority_class = max(class_dist.items(), key=lambda x: x[1])[0]
    majority_count = class_dist[majority_class]
    
    # Calculate target count for majority class
    target_count = int(majority_count * target_ratio)
    
    # Get all samples
    majority_samples = [item for item in data if 'Verdict: ' + majority_class in item['response']]
    minority_samples = [item for item in data if 'Verdict: ' + majority_class not in item['response']]
    
    # Randomly sample majority class
    sampled_majority = random.sample(majority_samples, target_count)
    
    # Combine with minority samples
    return sampled_majority + minority_samples

def augment_text(text: str, n_augmentations: int = 1) -> List[str]:
    """Augment text using synonym replacement."""
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    augmented_texts = []
    words = text.split()
    
    for _ in range(n_augmentations):
        new_words = words.copy()
        # Randomly select words to replace
        n_words_to_replace = max(1, len(words) // 10)  # Replace 10% of words
        words_to_replace = random.sample(range(len(words)), n_words_to_replace)
        
        for idx in words_to_replace:
            word = words[idx]
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
            
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        augmented_texts.append(' '.join(new_words))
    
    return augmented_texts

def augment_minority_classes(data: List[Dict[str, Any]], n_augmentations: int = 2) -> List[Dict[str, Any]]:
    """Augment minority classes using text augmentation."""
    class_dist = get_class_distribution(data)
    majority_class = max(class_dist.items(), key=lambda x: x[1])[0]
    
    augmented_data = data.copy()
    
    for label, count in class_dist.items():
        if label != majority_class:
            # Get samples of this class
            class_samples = [item for item in data if 'Verdict: ' + label in item['response']]
            
            for sample in class_samples:
                # Extract claim from instruction
                instruction = sample['instruction']
                claim_start = instruction.find('Claim: "') + len('Claim: "')
                claim_end = instruction.find('"\n', claim_start)
                claim = instruction[claim_start:claim_end]
                
                # Extract context
                context_start = instruction.find('Context: \n') + len('Context: \n')
                context_end = instruction.find('\n\nBased on', context_start)
                context = instruction[context_start:context_end]
                
                # Augment claim and context
                augmented_claims = augment_text(claim, n_augmentations)
                augmented_contexts = augment_text(context, n_augmentations)
                
                # Create new samples with augmented text
                for aug_claim, aug_context in zip(augmented_claims, augmented_contexts):
                    new_sample = sample.copy()
                    new_instruction = instruction[:claim_start] + aug_claim + instruction[claim_end:context_start] + aug_context + instruction[context_end:]
                    new_sample['instruction'] = new_instruction
                    augmented_data.append(new_sample)
    
    return augmented_data

def apply_class_weights(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate class weights for training."""
    class_dist = get_class_distribution(data)
    total_samples = sum(class_dist.values())
    
    # Calculate weights inversely proportional to class frequencies
    weights = {
        label: total_samples / (len(class_dist) * count)
        for label, count in class_dist.items()
    }
    
    return weights

def main():
    # Load training data
    print("Loading training data...")
    train_data = load_data('fake_news_model/data/processed/train_formatted.json')
    
    # Print original class distribution
    print("\nOriginal class distribution:")
    original_dist = get_class_distribution(train_data)
    total_original = sum(original_dist.values())
    for label, count in original_dist.items():
        percentage = (count / total_original) * 100
        print(f"{label}: {count} samples ({percentage:.1f}%)")
    
    # Apply balancing techniques
    print("\nApplying balancing techniques...")
    
    # 1. Oversampling
    print("\nOversampling minority classes...")
    oversampled_data = oversample_minority_classes(train_data)
    save_data(oversampled_data, 'fake_news_model/data/processed/train_oversampled.json')
    print("Oversampled class distribution:")
    oversampled_dist = get_class_distribution(oversampled_data)
    total_oversampled = sum(oversampled_dist.values())
    for label, count in oversampled_dist.items():
        percentage = (count / total_oversampled) * 100
        print(f"{label}: {count} samples ({percentage:.1f}%)")
    
    # 2. Undersampling
    print("\nUndersampling majority class...")
    undersampled_data = undersample_majority_class(train_data)
    save_data(undersampled_data, 'fake_news_model/data/processed/train_undersampled.json')
    print("Undersampled class distribution:")
    undersampled_dist = get_class_distribution(undersampled_data)
    total_undersampled = sum(undersampled_dist.values())
    for label, count in undersampled_dist.items():
        percentage = (count / total_undersampled) * 100
        print(f"{label}: {count} samples ({percentage:.1f}%)")
    
    # 3. Text Augmentation
    print("\nAugmenting minority classes...")
    augmented_data = augment_minority_classes(train_data)
    save_data(augmented_data, 'fake_news_model/data/processed/train_augmented.json')
    print("Augmented class distribution:")
    augmented_dist = get_class_distribution(augmented_data)
    total_augmented = sum(augmented_dist.values())
    for label, count in augmented_dist.items():
        percentage = (count / total_augmented) * 100
        print(f"{label}: {count} samples ({percentage:.1f}%)")
    
    # 4. Class Weights
    print("\nCalculating class weights...")
    class_weights = apply_class_weights(train_data)
    save_data(class_weights, 'fake_news_model/data/processed/class_weights.json')
    print("Class weights:")
    for label, weight in class_weights.items():
        print(f"{label}: {weight:.2f}")
    
    print("\nBalanced datasets have been saved to:")
    print("1. fake_news_model/data/processed/train_oversampled.json")
    print("2. fake_news_model/data/processed/train_undersampled.json")
    print("3. fake_news_model/data/processed/train_augmented.json")
    print("4. fake_news_model/data/processed/class_weights.json")

if __name__ == "__main__":
    main() 