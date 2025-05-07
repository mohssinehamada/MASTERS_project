import torch
import os
import re
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API", 
              description="API for detecting fake news using a fine-tuned Mistral model",
              version="1.0.0")

# Global model variables
model = None
tokenizer = None

class FactCheckRequest(BaseModel):
    claim: str
    evidence: Optional[str] = ""

class FactCheckResponse(BaseModel):
    verdict: str
    confidence: str
    reasoning: str
    sources: Optional[List[str]] = None

def extract_verdict_and_reasoning(response_text):
    """
    Extract verdict and reasoning from model response text
    """
    # Extract verdict
    verdict_pattern = r"Verdict:\s*(TRUE|FALSE|PARTIALLY TRUE|UNVERIFIABLE)"
    verdict_match = re.search(verdict_pattern, response_text, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "UNVERIFIABLE"
    
    # Extract reasoning
    reasoning_pattern = r"Reasoning:\s*(.*?)(?:\n\n|$)"
    reasoning_match = re.search(reasoning_pattern, response_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
    
    # Determine confidence
    if "insufficient evidence" in reasoning.lower() or "cannot determine" in reasoning.lower():
        confidence = "Low"
    elif "clearly" in reasoning.lower() or "definitely" in reasoning.lower() or "strongly" in reasoning.lower():
        confidence = "High"
    else:
        confidence = "Medium"
    
    return verdict, confidence, reasoning

def get_model():
    """
    Load model on first request
    """
    global model, tokenizer
    
    if model is None:
        model_path = os.getenv("MODEL_PATH", "./model/checkpoints/final")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Using base model instead.")
            model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    return model, tokenizer

@app.on_event("startup")
async def startup_event():
    """
    Pre-load model on startup if GPU is available
    """
    if torch.cuda.is_available():
        logger.info("GPU detected, pre-loading model")
        get_model()
    else:
        logger.info("No GPU detected, model will be loaded on first request")

@app.get("/")
def read_root():
    """API root endpoint"""
    return {"message": "Fake News Detection API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/analyze", response_model=FactCheckResponse)
async def analyze_claim(request: FactCheckRequest):
    """
    Analyze a claim for factual accuracy
    """
    try:
        # Load model and tokenizer
        model, tokenizer = get_model()
        
        # Format evidence
        evidence = request.evidence if request.evidence else "No additional context provided."
        
        # Create prompt
        prompt = f"""Analyze the following claim and determine if it is true or false:
        
Claim: {request.claim}

Context: {evidence}

Based on the provided context, perform a factual analysis of the claim.
Provide a verdict (TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE) and explain your reasoning.
"""

        # Generate response from model with truncation to match training configuration
        input_ids = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).input_ids.to(model.device)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract verdict and reasoning
        verdict, confidence, reasoning = extract_verdict_and_reasoning(response_text)
        
        # Extract sources if present in the evidence
        sources = []
        if "http" in evidence:
            urls = re.findall(r'https?://[^\s]+', evidence)
            sources = urls[:5]  # Limit to 5 sources
        
        logger.info(f"Processed claim: '{request.claim[:50]}...' - Verdict: {verdict}")
        
        return FactCheckResponse(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            sources=sources if sources else None
        )
    
    except Exception as e:
        logger.error(f"Error analyzing claim: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing claim: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 