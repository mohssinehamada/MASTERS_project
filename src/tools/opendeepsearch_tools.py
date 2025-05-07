try:
    from opendeepsearch import OpenDeepSearchTool
    OPENDEEPSEARCH_AVAILABLE = True
except ImportError:
    OPENDEEPSEARCH_AVAILABLE = False
    print("OpenDeepSearch not available. Using mock implementation.")

import os
from dotenv import load_dotenv
try:
    from loguru import logger
except ImportError:
    import logging as logger

# Load environment variables
load_dotenv()

# Setup mock or real search agent
if OPENDEEPSEARCH_AVAILABLE:
    # Initialize OpenDeepSearch with the correct configuration
    search_agent = OpenDeepSearchTool(
        model_name="openrouter/google/gemini-2.0-flash-001",  # Using the recommended model
        reranker="jina",  # Using Jina for reranking
        search_provider="serper"  # Using Serper as the search provider
    )
    
    # Set up the search agent
    search_agent.setup()

def search_with_opendeepsearch(query):
    """
    Perform a search using OpenDeepSearch instead of Google.
    Returns a list of relevant URLs and their content.
    """
    if not OPENDEEPSEARCH_AVAILABLE:
        # Return mock data for testing
        print(f"Mock search for: {query}")
        return [{
            'url': 'Mock Search Result',
            'content': f"This is a mock search result for: {query}\n\nHere are some example URLs:\nhttps://www.snopes.com/fact-check/\nhttps://www.factcheck.org/\nhttps://www.politifact.com/\nhttps://apnews.com/hub/ap-fact-check\nhttps://www.reuters.com/fact-check/"
        }]

    try:
        print(f"Using {search_agent.reranker} Reranker")
        # Forward the query to OpenDeepSearch and get the answer
        answer = search_agent.forward(query)
        
        # Check what type of response we got and handle accordingly
        if hasattr(answer, 'answer'):
            # Handle SearchResult object
            content = answer.answer
        elif hasattr(answer, 'get'):
            # Handle dictionary-like object
            content = answer.get('answer', str(answer))
        else:
            # Handle string or other types
            content = str(answer)
        
        # Format the result in the expected structure
        return [{
            'url': 'OpenDeepSearch Result',
            'content': content
        }]
    except Exception as e:
        if hasattr(logger, 'error'):
            logger.error(f"Error in OpenDeepSearch: {str(e)}")
        print(f"Error in OpenDeepSearch: {str(e)}")
        # Return a default response to prevent further errors
        return [{
            'url': 'Error',
            'content': f"Error occurred: {str(e)}"
        }] 