#!/usr/bin/env python
"""
Fake News Detector
Main entry point for running the fake news detection and benchmark tools
"""

import argparse
import sys
import os

# Add the project root to Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the modules directly
from src.agent.fake_news_detector import main as run_fake_news_detector
from src.benchmark.run_tests import main as run_tests

def main():
    parser = argparse.ArgumentParser(
        description='Fake News Detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Fake news detector command
    fakenews_parser = subparsers.add_parser('fakenews', help='Run the fake news detector')
    fakenews_parser.add_argument('claim', nargs='*', 
                              help='The claim to fact-check (if omitted, uses default example)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--mode', 
                              choices=['benchmark', 'compare', 'all'],
                              default='benchmark',
                              help='Test mode')
    benchmark_parser.add_argument('--quick', action='store_true', 
                              help='Run a quick benchmark (1 query, 1 website)')
    benchmark_parser.add_argument('--full', action='store_true', 
                              help='Run a full benchmark (5 queries, 3 websites each)')
    
    args = parser.parse_args()
    
    if args.command == 'fakenews':
        if args.claim:
            claim = ' '.join(args.claim)
            sys.argv = [sys.argv[0], claim]  # Pass claim to the detector
        run_fake_news_detector()
    elif args.command == 'benchmark':
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove 'benchmark' command for subprocess
        run_tests()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 