"""
Integration with Mistral-based fake news detection model
"""

import os
import requests
import json
from datetime import datetime

class ModelIntegration:
    def __init__(self, api_url=None):
        """
        Initialize the model integration
        
        Args:
            api_url: URL of the API endpoint (defaults to MODEL_API_URL env var)
        """
        self.api_url = api_url or os.getenv("MODEL_API_URL", "http://localhost:8000/analyze")
        self.enabled = self._check_api_available()
    
    def _check_api_available(self):
        """Check if the API is available"""
        try:
            # Try to connect to the health endpoint
            health_url = self.api_url.replace('/analyze', '/health')
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except:
            # If any error occurs, assume API is not available
            return False
    
    def analyze_claim(self, claim, evidence=""):
        """
        Analyze a claim using the model API
        
        Args:
            claim: The claim to analyze
            evidence: Optional evidence text
            
        Returns:
            Dictionary with analysis results in the expected format
        """
        if not self.enabled:
            return self._generate_fallback_result(claim, "Model API is not available")
        
        # Prepare request data
        data = {
            "claim": claim,
            "evidence": evidence
        }
        
        try:
            # Make request to API
            response = requests.post(self.api_url, json=data, timeout=30)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Format response in the expected format
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
            return self._generate_fallback_result(claim, f"Error connecting to model API: {str(e)}")
    
    def _generate_fallback_result(self, claim, reason):
        """Generate a fallback result when analysis can't be completed"""
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": "Low",
            "key_evidence": [reason],
            "competing_evidence": [],
            "reasoning": reason,
            "claim": claim,
            "timestamp": datetime.now().isoformat(),
            "sources": []
        }

# Factory function to get an instance of the model integration
def get_model_integration():
    """Get an instance of the model integration"""
    return ModelIntegration() 