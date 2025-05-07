from hyperbrowser import Browser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize HyperBrowser with API key
browser = Browser(
    api_key="hb_b696ec335d6f340ef953edaf9a90",
    project="mortgage_rate_scraper"
)

# Configure browser settings
browser.settings(
    max_steps=100,
    max_events=1000,
    sampling_rate=1.0,
    capture_errors=True,
    capture_metrics=True
)

def start_visualization():
    """Start the HyperBrowser visualization session"""
    browser.launch()
    return browser

def end_visualization():
    """End the HyperBrowser visualization session"""
    browser.close()

def add_step(step_name, data):
    """Add a step to the visualization"""
    browser.step(step_name, data)

def add_event(event_name, data):
    """Add an event to the visualization"""
    browser.event(event_name, data)

def capture_error(error_type, error_data):
    """Capture an error in the visualization"""
    browser.error(error_type, error_data)

def capture_metric(metric_name, value):
    """Capture a metric in the visualization"""
    browser.metric(metric_name, value) 