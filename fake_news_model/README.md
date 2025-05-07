# Fake News Detection Model

A Mistral-based model for detecting fake news and fact checking claims.

## Project Structure

```
fake_news_model/
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed training data
│   └── evaluation/           # Test datasets
├── model/
│   ├── training/             # Training scripts
│   ├── inference/            # Inference code
│   └── checkpoints/          # Saved model checkpoints
├── api/
│   ├── server.py             # FastAPI server
│   └── routes.py             # API endpoints
├── utils/
│   ├── preprocessing.py      # Data preprocessing
│   └── evaluation.py         # Evaluation metrics
├── requirements.txt          # Dependencies
├── Dockerfile                # For containerization
└── README.md                 # Documentation
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download pre-processed datasets:
   ```
   python -m utils.download_datasets
   ```

3. Fine-tune the model (requires GPU with 16GB+ VRAM):
   ```
   python -m model.training.train
   ```

4. Start the inference API:
   ```
   python -m api.server
   ```

5. Update the environment variable in your fake news agent:
   ```
   MODEL_API_URL=http://localhost:8000/analyze
   ```

## Using Docker

1. Build the container:
   ```
   docker build -t fake-news-model .
   ```

2. Run with GPU support:
   ```
   docker run --gpus all -p 8000:8000 fake-news-model
   ```

## API Usage

Send POST requests to `/analyze` with JSON payload:
```json
{
  "claim": "Your claim text here",
  "evidence": "Optional supporting evidence text"
}
```

Response format:
```json
{
  "verdict": "TRUE|FALSE|PARTIALLY TRUE|UNVERIFIABLE",
  "confidence": "High|Medium|Low",
  "reasoning": "Detailed explanation of the analysis..."
}
```

## Integration with Fake News Agent

To integrate this model with your existing fake news detection agent, update the environment variables to point to this API instead of using external services.

## Model Details

- Base Model: Mistral 7B Instruct v0.2
- Fine-tuning: LoRA adaptation on fact-checking datasets
- Quantization: 4-bit for efficient inference
- Performance: Optimized for accuracy in fake news detection tasks

## License

This project is for educational and research purposes only. 