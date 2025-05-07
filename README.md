# Fake News Detection with DistilGPT2 and LoRA

A machine learning-based system for detecting fake news using DistilGPT2 and LoRA fine-tuning. This project is part of the MASTERS research work on fact-checking and misinformation detection.

## Project Overview

The project implements a sophisticated fake news detection system using state-of-the-art transformer models. It uses DistilGPT2 as the base model with LoRA (Low-Rank Adaptation) fine-tuning for efficient training.

### Key Features

* Implementation of DistilGPT2 with LoRA for fake news detection
* Multiple data balancing techniques:
  - Oversampling
  - Undersampling
  - Text augmentation
  - Class weights
* Comprehensive evaluation metrics
* Training visualization and logging
* Combined dataset approach for better performance

## Project Structure

```
fake_news_agent/
├── fake_news_model/
│   ├── data/
│   │   ├── raw/                 # Raw dataset files
│   │   └── processed/           # Processed and balanced datasets
│   ├── model/
│   │   ├── training/           # Training scripts
│   │   ├── checkpoints/        # Model checkpoints
│   │   ├── logs/              # Training logs
│   │   ├── models/            # Saved models
│   │   ├── visualizations/    # Training visualizations
│   │   └── evaluation/        # Model evaluation results
│   └── utils/                 # Utility functions
└── requirements.txt           # Project dependencies
```

## Dataset Structure

The dataset is split into three parts:
* Training set: 8,873 examples
* Validation set: 986 examples
* Test set: 2,465 examples

Label Distribution:
* TRUE
* FALSE
* PARTIALLY TRUE

Sources:
* LIAR dataset
* PolitiFact

## Model Architecture

- Base model: DistilGPT2
- Fine-tuning: LoRA (Low-Rank Adaptation)
- Target modules: ["c_attn", "c_proj"]
- LoRA configuration:
  - r: 8
  - alpha: 32
  - dropout: 0.05

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mohssinehamada/MASTERS_project.git
cd MASTERS_project
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the required format:
```json
{
    "instruction": "Your news text here",
    "response": "TRUE/FALSE/PARTIALLY TRUE"
}
```

2. Place your data in the appropriate directories:
   - Raw data: `fake_news_model/data/raw/`
   - Processed data: `fake_news_model/data/processed/`

3. Run the training script:
```bash
python fake_news_model/model/training/train.py
```

## Evaluation

The model is evaluated using:
- Confusion matrix
- Classification report
- Per-class metrics
- Training loss visualization

## Output

The training process generates:
1. Model checkpoints
2. Training logs
3. Visualizations:
   - Training loss plot
   - Confusion matrix
   - Per-class metrics
4. Evaluation metrics in JSON format

## Dependencies

- transformers==4.36.0
- accelerate==0.25.0
- peft==0.7.1
- torch>=1.13.0
- datasets
- nltk
- pandas
- matplotlib
- seaborn
- scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformers library
- The DistilGPT2 model authors
- The LoRA paper authors
