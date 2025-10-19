#!/usr/bin/env python3
"""
Plot benchmark results similar to Figure 3 from RedHat article.
Shows TPOT (Time Per Output Token) vs QPS for different model configurations.
"""

import json
import glob
import matplotlib.pyplot as plt
import os
import re

def load_benchmark_results(logs_dir="logs/json"):
    """Load all benchmark results from JSON files."""
    results = {}
    
    # Find all result directories
    for result_dir in glob.glob(f"{logs_dir}/*"):
        if not os.path.isdir(result_dir):
            continue
            
        # Parse directory name to extract model config and QPS
        dir_name = os.path.basename(result_dir)
        print(f"Processing directory: {dir_name}")
        
        # Try multiple naming patterns
        model_name = None
        qps = None
        
        # Pattern 1: W8A8_1_1755477435 (your actual format)
        match = re.match(r'([^_]+)_(\d+)_\d+', dir_name)
        if match:
            model_name = match.group(1)
            qps = int(match.group(2))
        else:
            # Pattern 2: {model_name}_qps_{qps}_{timestamp} (original expected format)
            match = re.match(r'(.+)_qps_(\d+)_\d+', dir_name)
            if match:
                model_name = match.group(1)
                qps = int(match.group(2))
        
        if not model_name or qps is None:
            print(f"  Could not parse directory name: {dir_name}")
            continue
            
        print(f"  Found: model={model_name}, qps={qps}")
        
        # Look for the results JSON file
        json_files = glob.glob(f"{result_dir}/*.json")
        if not json_files:
            print(f"  No JSON files found in {result_dir}")
            continue
            
        # Load the results
        try:
            json_file = json_files[0]  # Use the first JSON file found
            print(f"  Loading: {os.path.basename(json_file)}")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if model_name not in results:
                results[model_name] = {}
                
            results[model_name][qps] = data
            print(f"  Successfully loaded data for {model_name} at {qps} QPS")
            
        except Exception as e:
            print(f"  Error loading {json_files[0]}: {e}")
            continue
    
    return results

def plot_tpot_vs_qps(results, output_file="benchmark_comparison.png"):
    """Plot TPOT vs QPS similar to Figure 3."""
    
    plt.figure(figsize=(10, 6))
    
    # Colors and markers for different model configs
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (model_name, model_data) in enumerate(results.items()):
        qps_values = []
        tpot_values = []
        ttft_values = []
        
        # Sort by QPS for proper line plotting
        for qps in sorted(model_data.keys()):
            data = model_data[qps]
            
            # Extract TPOT (Time Per Output Token) - convert from ms to seconds
            tpot = None
            if 'mean_tpot_ms' in data:
                tpot = data['mean_tpot_ms'] / 1000.0  # Convert ms to seconds
            elif 'tpot' in data:
                tpot = data['tpot']
                
            if tpot is not None:
                qps_values.append(qps)
                tpot_values.append(tpot)
                
                # Also track TTFT for the 5-second constraint - convert from ms to seconds
                ttft = None
                if 'mean_ttft_ms' in data:
                    ttft = data['mean_ttft_ms'] / 1000.0  # Convert ms to seconds
                elif 'ttft' in data:
                    ttft = data['ttft']
                    
                if ttft is not None:
                    ttft_values.append(ttft)
        
        if qps_values:
            plt.plot(qps_values, tpot_values, 
                    color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)],
                    linewidth=2, markersize=8,
                    label=f"{model_name}")
    
    # Add horizontal line at 5 seconds (TTFT constraint from article)
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.7, 
                label='5s TTFT limit')
    
    plt.xlabel('Query Rate (QPS)', fontsize=12)
    plt.ylabel('Time Per Output Token (seconds)', fontsize=12)
    plt.title('Model Performance Comparison: TPOT vs QPS', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(0, max([max(model_data.keys()) for model_data in results.values()]) + 1)
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()

def print_summary_table(results):
    """Print a summary table of results."""
    print("\n=== Benchmark Results Summary ===")
    print(f"{'Model':<15} {'QPS':<5} {'TPOT(s)':<8} {'TTFT(s)':<8} {'Throughput':<12}")
    print("-" * 55)
    
    for model_name, model_data in results.items():
        for qps in sorted(model_data.keys()):
            data = model_data[qps]
            
            # Extract TPOT (convert from ms to seconds)
            tpot = 'N/A'
            if 'mean_tpot_ms' in data:
                tpot = f"{data['mean_tpot_ms'] / 1000.0:.3f}"
            elif 'tpot' in data:
                tpot = f"{data['tpot']:.3f}"
            
            # Extract TTFT (convert from ms to seconds)
            ttft = 'N/A'
            if 'mean_ttft_ms' in data:
                ttft = f"{data['mean_ttft_ms'] / 1000.0:.3f}"
            elif 'ttft' in data:
                ttft = f"{data['ttft']:.3f}"
            
            # Extract throughput
            throughput = 'N/A'
            if 'request_throughput' in data:
                throughput = f"{data['request_throughput']:.2f}"
                
            print(f"{model_name:<15} {qps:<5} {tpot:<8} {ttft:<8} {throughput:<12}")

if __name__ == "__main__":
    # Load results
    results = load_benchmark_results()
    
    if not results:
        print("No benchmark results found in logs/json/")
        print("Make sure you've run the benchmark script first!")
        exit(1)
    
    print(f"Found results for {len(results)} model configurations:")
    for model_name, model_data in results.items():
        print(f"  {model_name}: {len(model_data)} QPS points")
    
    # Plot the results
    plot_tpot_vs_qps(results)
    
    # Print summary
    print_summary_table(results)