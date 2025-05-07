import json
import os
from datetime import datetime

def save_results(data, data_type):
    """
    Save results to a JSON file with timestamp.
    
    Args:
        data: The data to save
        data_type: The type of data (used in filename)
        
    Returns:
        The path where the data was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create filename
    filename = f"data/results_{data_type}_{timestamp}.json"
    
    # Save data to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filename

def format_duration(seconds):
    """
    Format duration in seconds to a readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    minutes, seconds = divmod(int(seconds), 60)
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s" 