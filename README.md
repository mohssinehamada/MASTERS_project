# Fact-Checking with BERT and RoBERTa

This project implements fact-checking using BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa (Robustly Optimized BERT Pretraining Approach) models. The system is designed to classify statements as either true or false based on sophisticated natural language processing techniques.

## Project Overview

The project uses state-of-the-art transformer models to perform binary classification on statements, determining their veracity. It implements both BERT and RoBERTa models for comparison and evaluation purposes.

### Key Features

- Implementation of BERT and RoBERTa models for fact-checking
- Binary classification (true/false) of statements
- Data preprocessing and tokenization
- Model training and evaluation
- Performance metrics calculation

## Technical Details

### Models Used

1. **BERT**
   - Base model: bert-base-uncased
   - Number of labels: 2 (binary classification)
   - Max sequence length: 256

2. **RoBERTa**
   - Base model: roberta-base
   - Number of labels: 2 (binary classification)
   - Max sequence length: 256

### Dataset Structure

The dataset is split into three parts:
- Training set
- Validation set
- Test set

Each entry contains:
- Statement text
- Label (true/false)
- Metadata (speaker, context, etc.)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/mohssinehamada/MASTERS_project.git
cd MASTERS_project
```

2. Install required packages:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy
```

## Model Performance

The BERT model achieved:
- Training accuracy: ~75%
- Validation accuracy: ~69%
- Test accuracy: ~75%

Performance metrics include:
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Project Structure

```
MASTERS_project/
├── BERT_Transformer_Models/
│   └── BERT_ROBERT.ipynb
├── README.md
└── .gitignore
```

## Usage

The main implementation is in the Jupyter notebook `BERT_Transformer_Models/BERT_ROBERT.ipynb`. To use the models:

1. Open the notebook in Jupyter Lab/Notebook
2. Follow the cells in sequence
3. The notebook includes:
   - Data preprocessing
   - Model initialization
   - Training
   - Evaluation
   - Testing

## Future Improvements

- Implementation of additional transformer models
- Hyperparameter optimization
- Ensemble methods
- Cross-validation
- Error analysis

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.