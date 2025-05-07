import requests
import argparse
import os
import sys
import json
from datetime import datetime

def analyze_claim_with_api(claim, evidence="", api_url=None):
    """
    Analyze a claim using the API
    
    Args:
        claim: The claim to analyze
        evidence: Optional evidence text
        api_url: URL of the API endpoint
        
    Returns:
        Dictionary with analysis results
    """
    # Get API URL from environment or use default
    if api_url is None:
        api_url = os.getenv("MODEL_API_URL", "http://localhost:8000/analyze")
    
    # Prepare request data
    data = {
        "claim": claim,
        "evidence": evidence
    }
    
    try:
        # Make request to API
        response = requests.post(api_url, json=data, timeout=30)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Format for the fake news detector agent
        formatted_result = {
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "key_evidence": [result["reasoning"]],
            "competing_evidence": [],
            "reasoning": result["reasoning"],
            "claim": claim,
            "timestamp": datetime.now().isoformat(),
            "sources": [{"domain": source, "url": source} for source in (result.get("sources") or [])]
        }
        
        return formatted_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        
        # Return a fallback response
        fallback = {
            "verdict": "UNVERIFIABLE",
            "confidence": "Low",
            "key_evidence": [f"Error connecting to model API: {str(e)}"],
            "competing_evidence": [],
            "reasoning": "Could not connect to the model API for analysis.",
            "claim": claim,
            "timestamp": datetime.now().isoformat(),
            "sources": []
        }
        
        return fallback

def main():
    """
    CLI for testing the integration
    """
    parser = argparse.ArgumentParser(description="Fake News Detection API Integration")
    parser.add_argument("--claim", required=True, help="The claim to analyze")
    parser.add_argument("--evidence", default="", help="Optional evidence text")
    parser.add_argument("--api_url", default=None, 
                        help="API URL (defaults to MODEL_API_URL env var or http://localhost:8000/analyze)")
    parser.add_argument("--output", default=None, help="Output file to write results (JSON)")
    
    args = parser.parse_args()
    
    # Analyze claim
    result = analyze_claim_with_api(args.claim, args.evidence, args.api_url)
    
    # Write to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        # Print results
        print("\n" + "="*50)
        print(f"CLAIM: {result['claim']}")
        print(f"VERDICT: {result['verdict']} (Confidence: {result['confidence']})")
        print("-"*50)
        print(f"REASONING: {result['reasoning']}")
        print("="*50)

if __name__ == "__main__":
    main() 