import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from fake_news_model.utils.preprocessing import load_and_prepare_datasets, create_prompt_response_pairs
import pandas as pd
from datasets import Dataset
import nltk
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def setup_nltk():
    """Setup NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def prepare_data_for_training(data, tokenizer):
    """Prepare data for training by tokenizing and formatting"""
    formatted_data = []
    for item in data:
        # Create the prompt
        prompt = f"User: {item['instruction']}\nAssistant: {item['response']}"
        
        # Tokenize the prompt
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Add to formatted data
        formatted_data.append({
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0].clone()
        })
    
    return formatted_data

def generate_dataset():
    """Generate combined dataset from all processed files"""
    print("Loading all datasets...")
    
    # Define file paths
    datasets = {
        "original": "fake_news_model/data/processed/train_formatted.json",
        "oversampled": "fake_news_model/data/processed/train_oversampled.json",
        "undersampled": "fake_news_model/data/processed/train_undersampled.json",
        "augmented": "fake_news_model/data/processed/train_augmented.json"
    }
    
    val_file = "fake_news_model/data/processed/validation_formatted.json"
    class_weights_file = "fake_news_model/data/processed/class_weights.json"
    
    # Load class weights if they exist
    class_weights = None
    if os.path.exists(class_weights_file):
        with open(class_weights_file, 'r') as f:
            class_weights = json.load(f)
            print("Loaded class weights:", class_weights)
    
    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            print(f"Loading {dataset_name} dataset...")
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_train_data.extend(data)
                print(f"Added {len(data)} examples from {dataset_name} dataset")
    
    # Load validation dataset
    if os.path.exists(val_file):
        print("Loading validation dataset...")
        with open(val_file, 'r') as f:
            eval_data = json.load(f)
    else:
        print("Validation dataset not found.")
        raise FileNotFoundError(f"Could not find validation dataset: {val_file}")
    
    print(f"Combined training dataset size: {len(combined_train_data)} examples")
    print(f"Validation dataset size: {len(eval_data)} examples")
    
    return combined_train_data, eval_data, class_weights

def plot_training_metrics(trainer, output_dir):
    """Plot training metrics"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get training history
    history = trainer.state.log_history
    
    # Extract metrics
    train_loss = [x.get('loss') for x in history if 'loss' in x]
    eval_loss = [x.get('eval_loss') for x in history if 'eval_loss' in x]
    steps = [x.get('step') for x in history if 'loss' in x]
    
    # Plot training and evaluation loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label='Training Loss')
    if eval_loss:
        plt.plot(steps, eval_loss, label='Evaluation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

def evaluate_model(model, tokenizer, eval_data, output_dir):
    """Evaluate model and create visualizations"""
    model.eval()
    predictions = []
    true_labels = []
    
    print("Evaluating model...")
    for example in tqdm(eval_data):
        # Create prompt
        prompt = f"User: {example['instruction']}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract prediction and true label
        pred_label = extract_label(response)
        true_label = example['response'].split()[0]  # Assuming first word is the label
        
        predictions.append(pred_label)
        true_labels.append(true_label)
    
    # Create confusion matrix
    labels = sorted(list(set(true_labels + predictions)))
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(true_labels, predictions, labels=labels, output_dict=True)
    
    # Plot per-class metrics
    metrics = ['precision', 'recall', 'f1-score']
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report[label][metric] for label in labels]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-class Metrics')
    plt.xticks(x + width, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'))
    plt.close()
    
    # Save detailed metrics
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)

def extract_label(response):
    """Extract label from model response"""
    # Simple extraction - can be made more robust
    words = response.split()
    for word in words:
        if word in ['TRUE', 'FALSE', 'PARTIALLY']:
            if word == 'PARTIALLY':
                return 'PARTIALLY TRUE'
            return word
    return 'UNKNOWN'

def load_checkpoint(checkpoint_path):
    """Load model and tokenizer from checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        return model, tokenizer
    return None, None

def save_checkpoint(model, tokenizer, checkpoint_path):
    """Save model and tokenizer checkpoint"""
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def train(use_class_weights=False):
    """Train the model with LoRA fine-tuning
    
    Args:
        use_class_weights (bool): Whether to use class weights for training
    """
    # Setup NLTK for any text processing
    setup_nltk()
    
    # Generate combined dataset
    train_data, eval_data, class_weights = generate_dataset()
    
    print(f"Training with {len(train_data)} examples, validating with {len(eval_data)} examples")
    if use_class_weights and class_weights:
        print("Using class weights:", class_weights)
    
    # Load model and tokenizer
    model_name = "distilgpt2"  # Smaller model, more suitable for CPU training
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data for training
    print("Preparing training data...")
    train_data = prepare_data_for_training(train_data, tokenizer)
    print("Preparing validation data...")
    eval_data = prepare_data_for_training(eval_data, tokenizer)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # Check if CUDA is available
    device = "cpu"  # Force CPU only for stability
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": device},
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # Correct attention modules for DistilGPT2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Create output directories
    os.makedirs("fake_news_model/checkpoints/fake_news_combined", exist_ok=True)
    os.makedirs("fake_news_model/logs/fake_news_combined", exist_ok=True)
    os.makedirs("fake_news_model/models/fake_news_combined", exist_ok=True)
    os.makedirs("fake_news_model/visualizations/fake_news_combined", exist_ok=True)
    os.makedirs("fake_news_model/evaluation/fake_news_combined", exist_ok=True)
    
    # Set up training arguments with checkpointing
    training_args = TrainingArguments(
        output_dir="fake_news_model/checkpoints/fake_news_combined",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        logging_dir="fake_news_model/logs/fake_news_combined",
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        group_by_length=True,
        lr_scheduler_type="constant",
        no_cuda=True,  # Force CPU training
        dataloader_num_workers=0,  # Disable multiprocessing for CPU training
        dataloader_pin_memory=False,  # Disable pin memory for CPU training
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Add class weights to training arguments if specified
    if use_class_weights and class_weights:
        training_args.class_weights = class_weights
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    output_dir = "fake_news_model/models/fake_news_combined"
    trainer.save_model(output_dir)
    
    # Plot training metrics
    plot_training_metrics(trainer, "fake_news_model/visualizations/fake_news_combined")
    
    # Evaluate the model
    evaluate_model(model, tokenizer, eval_data, "fake_news_model/evaluation/fake_news_combined")
    
    print(f"Training completed. Model saved to {output_dir}")
    return model, tokenizer

def train_all_configurations():
    """Train the model with all dataset configurations"""
    configurations = [
        (False,),  # Without class weights
        (True,)    # With class weights
    ]
    
    results = {}
    for use_weights in configurations:
        print(f"\n{'='*50}")
        print(f"Training with combined dataset" + 
              (" and class weights" if use_weights[0] else ""))
        print(f"{'='*50}\n")
        
        try:
            model, tokenizer = train(use_class_weights=use_weights[0])
            results[f"combined_{'with_weights' if use_weights[0] else 'no_weights'}"] = "Success"
        except Exception as e:
            print(f"Error training with combined dataset: {str(e)}")
            results[f"combined_{'with_weights' if use_weights[0] else 'no_weights'}"] = f"Failed: {str(e)}"
    
    # Print summary of results
    print("\nTraining Summary:")
    print("="*50)
    for config, status in results.items():
        print(f"{config}: {status}")
    print("="*50)

if __name__ == "__main__":
    train_all_configurations() 