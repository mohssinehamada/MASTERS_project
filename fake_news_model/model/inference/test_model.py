import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re
import time
import json
import numpy as np

def extract_verdict_and_reasoning(response_text):
    """
    Extract verdict and reasoning from model response text
    """
    # Extract verdict
    verdict_pattern = r"Verdict:\s*(TRUE|FALSE|PARTIALLY TRUE|UNVERIFIABLE)"
    verdict_match = re.search(verdict_pattern, response_text, re.IGNORECASE)
    
    # If no verdict found in expected format, try to infer from the text
    if not verdict_match:
        text_lower = response_text.lower()
        if "true" in text_lower and not any(x in text_lower for x in ["partially", "not true", "false"]):
            verdict = "TRUE"
        elif "false" in text_lower and not "not false" in text_lower:
            verdict = "FALSE"
        elif "partially" in text_lower:
            verdict = "PARTIALLY TRUE"
        else:
            verdict = "UNVERIFIABLE"
    else:
        verdict = verdict_match.group(1).upper()
    
    # Extract reasoning
    reasoning_pattern = r"Reasoning:\s*(.*?)(?:\n\n|$)"
    reasoning_match = re.search(reasoning_pattern, response_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
    
    # Determine confidence and score based on reasoning and evidence strength
    confidence_scores = {
        'TRUE': 0.1,
        'FALSE': 0.1,
        'PARTIALLY TRUE': 0.1,
        'UNVERIFIABLE': 0.1
    }
    
    # Analyze reasoning for evidence strength
    evidence_strength = 0.0
    strong_evidence = ["scientific evidence", "multiple studies", "clear evidence", "proven", "verified"]
    moderate_evidence = ["suggests", "indicates", "likely", "research shows"]
    weak_evidence = ["might", "could", "possibly", "unclear", "insufficient evidence"]
    
    text_lower = reasoning.lower()
    
    if any(phrase in text_lower for phrase in strong_evidence):
        evidence_strength = 0.8
    elif any(phrase in text_lower for phrase in moderate_evidence):
        evidence_strength = 0.6
    elif any(phrase in text_lower for phrase in weak_evidence):
        evidence_strength = 0.3
    else:
        evidence_strength = 0.5
    
    # Set confidence based on evidence strength
    if evidence_strength >= 0.7:
        confidence = "High"
        base_score = 0.8
    elif evidence_strength >= 0.4:
        confidence = "Medium"
        base_score = 0.6
    else:
        confidence = "Low"
        base_score = 0.4
    
    # Adjust scores based on verdict and evidence
    confidence_scores[verdict] = base_score
    
    # If evidence is weak, increase UNVERIFIABLE score
    if evidence_strength < 0.4:
        confidence_scores['UNVERIFIABLE'] = max(confidence_scores['UNVERIFIABLE'], 0.4)
    
    # Normalize scores to sum to 1.0
    total = sum(confidence_scores.values())
    for k in confidence_scores:
        confidence_scores[k] /= total
    
    # Convert scores to list in fixed order
    score_list = [
        confidence_scores['TRUE'],
        confidence_scores['FALSE'],
        confidence_scores['PARTIALLY TRUE'],
        confidence_scores['UNVERIFIABLE']
    ]
    
    return verdict, confidence, reasoning, score_list

def load_model(model_path):
    """
    Load the model and tokenizer
    """
    print(f"Loading model from {model_path}")
    
    try:
        # Try loading as a regular model first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model = model.to("cpu")
        
        print("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to base model...")
        
        # Fallback to distilgpt2
        model_path = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model = model.to("cpu")
        
        print("Fallback model loaded successfully")
        return model, tokenizer

def analyze_claim(model, tokenizer, claim, evidence=""):
    """
    Analyze a claim using the model
    """
    # Format evidence
    evidence = evidence if evidence else "No additional context provided."
    
    # Create prompt
    prompt = f"""Analyze the following claim and determine if it is true or false:
    
Claim: {claim}

Context: {evidence}

Based on the provided context, perform a factual analysis of the claim.
You must provide a clear verdict from one of these options: TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE.
Choose TRUE if the claim is completely accurate.
Choose FALSE if the claim is completely inaccurate.
Choose PARTIALLY TRUE if the claim has some truth but is not entirely accurate.
Choose UNVERIFIABLE if there is insufficient evidence to determine the truth.

Provide your verdict and detailed reasoning below:

Verdict: """

    # Generate response from model
    start_time = time.time()
    
    # Tokenize with proper attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
        return_attention_mask=True
    ).to("cpu")
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Extract verdict and reasoning
    verdict, confidence, reasoning, prediction_scores = extract_verdict_and_reasoning(response_text)
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    result = {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "generation_time": f"{generation_time:.2f} seconds",
        "prediction_scores": prediction_scores
    }
    
    return result, response_text

def save_results(results, output_file):
    """Save results to a JSON file for visualization"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert results to format needed for visualization
    viz_data = {
        'true_labels': [],
        'predictions': [],
        'prediction_scores': [],
        'confidences': []
    }
    
    for result in results:
        viz_data['predictions'].append(result['verdict'])
        viz_data['prediction_scores'].append(result['prediction_scores'])
        viz_data['confidences'].append(float(result['confidence'].lower() == 'high'))
    
    with open(output_file, 'w') as f:
        json.dump(viz_data, f)

def main():
    parser = argparse.ArgumentParser(description="Test fake news detection model")
    parser.add_argument("--model_path", default="fake_news_model/model/checkpoints/checkpoint-831", help="Path to the model")
    parser.add_argument("--claim", default="The Earth is flat", help="Claim to analyze")
    parser.add_argument("--evidence", default="", help="Evidence to provide (optional)")
    parser.add_argument("--test_file", default="", help="Path to test file with multiple claims")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    results = []
    
    if args.test_file:
        # Process multiple claims from test file
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
            for item in test_data:
                claim = item['claim']
                evidence = item.get('evidence', '')
                true_label = item.get('true_label', 'UNVERIFIABLE')
                
                result, _ = analyze_claim(model, tokenizer, claim, evidence)
                result['true_label'] = true_label  # Add true label to results
                results.append(result)
                
                # Print results for each claim
                print("\n" + "="*50)
                print(f"CLAIM: {result['claim']}")
                print(f"TRUE LABEL: {true_label}")
                print(f"PREDICTED: {result['verdict']} (Confidence: {result['confidence']})")
                print("-"*50)
                print(f"REASONING: {result['reasoning']}")
                print("-"*50)
                print(f"Generation time: {result['generation_time']}")
                print("="*50)
    else:
        # Process single claim
        result, full_response = analyze_claim(model, tokenizer, args.claim, args.evidence)
        results.append(result)
        
        # Print results
        print("\n" + "="*50)
        print(f"CLAIM: {result['claim']}")
        print(f"VERDICT: {result['verdict']} (Confidence: {result['confidence']})")
        print("-"*50)
        print(f"REASONING: {result['reasoning']}")
        print("-"*50)
        print(f"Generation time: {result['generation_time']}")
        print("="*50)
    
    # Save results for visualization
    save_results(results, 'fake_news_model/model/checkpoints/metrics/test_results.json')
    
    # Sample claims to test if no claim provided
    if args.claim == "The Earth is flat" and not args.test_file:
        print("\nHere are some other claims you can test:")
        sample_claims = [
            "COVID-19 vaccines contain microchips",
            "Drinking water with lemon every morning cures cancer",
            "Global temperatures have risen by 1.1°C since pre-industrial times",
            "5G towers cause COVID-19",
            "Vaccines cause autism"
        ]
        
        for claim in sample_claims:
            print(f"- {claim}")
        
        print("\nRun with: python -m model.inference.test_model --claim \"YOUR CLAIM HERE\"")

if __name__ == "__main__":
    main() 