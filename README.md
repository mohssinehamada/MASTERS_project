# Fake News Detection Agent

A machine learning-based system for detecting fake news using DistilGPT2 and LoRA fine-tuning.

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

## Features

- Data balancing techniques:
  - Oversampling
  - Undersampling
  - Text augmentation
  - Class weights
- LoRA fine-tuning for efficient model training
- Combined dataset training approach
- Comprehensive evaluation metrics
- Training visualization and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake_news_agent.git
cd fake_news_agent
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

## Data Preparation

The project uses several data balancing techniques to handle class imbalance:

1. Original dataset
2. Oversampled dataset
3. Undersampled dataset
4. Augmented dataset

All datasets are combined for training to maximize the available data.

## Training

The model is trained using LoRA fine-tuning on DistilGPT2. To start training:

```bash
python fake_news_model/model/training/train.py
```

Training configurations:
- Model: DistilGPT2
- Training approach: LoRA fine-tuning
- Batch size: 4
- Learning rate: 2e-4
- Number of epochs: 3
- Device: CPU (configurable for GPU)

## Model Architecture

- Base model: DistilGPT2
- Fine-tuning: LoRA (Low-Rank Adaptation)
- Target modules: ["c_attn", "c_proj"]
- LoRA configuration:
  - r: 8
  - alpha: 32
  - dropout: 0.05

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

## Results

The model is trained on a combined dataset that includes:
- Original data
- Oversampled data
- Undersampled data
- Augmented data

This approach maximizes the available training data and helps handle class imbalance.

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