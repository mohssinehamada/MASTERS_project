#!/usr/bin/env python
import os
import sys
import time
import json
from datetime import datetime
import subprocess
import argparse

def run_benchmark(queries=2, websites=2, timeout=30):
    """Run the benchmark with specific parameters"""
    print(f"\n=== Running benchmark: {queries} queries, {websites} websites, {timeout}s timeout ===\n")
    
    cmd = f"python -m src.benchmark.benchmark --queries {queries} --websites {websites} --timeout {timeout}"
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        if not line:
            break
    
    # Wait for process to complete
    process.wait()
    
    # Check for errors
    if process.returncode != 0:
        print("Error in benchmark execution:")
        for line in process.stderr:
            print(line, end='')
    
    duration = time.time() - start_time
    print(f"\nBenchmark completed in {duration:.2f} seconds")
    
    return process.returncode == 0

def run_agent():
    """Run the standard agent"""
    print("\n=== Running standard agent ===\n")
    
    cmd = "python -m src.agent.agent"
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        if not line:
            break
    
    # Wait for process to complete
    process.wait()
    
    # Check for errors
    if process.returncode != 0:
        print("Error in agent execution:")
        for line in process.stderr:
            print(line, end='')
    
    duration = time.time() - start_time
    print(f"\nAgent completed in {duration:.2f} seconds")
    
    return process.returncode == 0

def compare_benchmarks():
    """Find and compare benchmark reports"""
    # Look for benchmark reports in the data directory
    if not os.path.exists("data"):
        print("No data directory found")
        return
        
    report_files = [os.path.join("data", f) for f in os.listdir("data") 
                   if f.startswith("benchmark_report_") and f.endswith(".json")]
    
    if not report_files:
        print("No benchmark reports found to compare")
        return
    
    # Sort by timestamp (newest first)
    report_files.sort(reverse=True)
    
    reports = []
    for file in report_files[:5]:  # Compare up to 5 most recent reports
        try:
            with open(file, 'r') as f:
                reports.append(json.load(f))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not reports:
        return
    
    print("\n=== Benchmark Comparison ===\n")
    print(f"{'Timestamp':<20} {'Queries':<8} {'Success':<8} {'Websites':<10} {'Rates':<8} {'Time (s)':<10}")
    print("-" * 70)
    
    for report in reports:
        summary = report['summary']
        timestamp = report['timestamp']
        formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
        
        print(f"{formatted_time:<20} "
              f"{summary['total_queries']:<8} "
              f"{summary['query_success_rate']:.1f}%{'':<2} "
              f"{summary['website_success_rate']:.1f}%{'':<5} "
              f"{summary['avg_rates_per_query']:.1f}{'':<5} "
              f"{summary['avg_execution_time']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Run agent tests and benchmarks')
    parser.add_argument('--mode', choices=['agent', 'benchmark', 'compare', 'all'], default='benchmark',
                      help='Which test to run: agent, benchmark, compare results, or all')
    parser.add_argument('--quick', action='store_true', help='Run a quick benchmark (1 query, 1 website)')
    parser.add_argument('--full', action='store_true', help='Run a full benchmark (all 5 queries, 3 websites each)')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick benchmark
        if args.mode in ['benchmark', 'all']:
            run_benchmark(queries=1, websites=1, timeout=20)
    elif args.full:
        # Full benchmark
        if args.mode in ['benchmark', 'all']:
            run_benchmark(queries=5, websites=3, timeout=40)
    else:
        # Default benchmark
        if args.mode in ['benchmark', 'all']:
            run_benchmark(queries=2, websites=2, timeout=30)
    
    # Run the agent if requested
    if args.mode in ['agent', 'all']:
        run_agent()
    
    # Compare results if requested
    if args.mode in ['compare', 'all']:
        compare_benchmarks()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 