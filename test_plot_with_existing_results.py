#!/usr/bin/env python3
"""
Test the plotting script with your existing W8A8 results.
"""

import os
import sys

# Add the current directory to Python path so we can import our script
sys.path.insert(0, '.')

from plot_benchmark_results import load_benchmark_results, plot_tpot_vs_qps, print_summary_table

def main():
    print("Testing plot script with existing results...")
    
    # Try to load results from the default location
    results = load_benchmark_results("logs/json")
    
    if not results:
        print("\n❌ No results found!")
        print("Make sure your results are in the logs/json/ directory")
        print("Current working directory:", os.getcwd())
        
        # Show what directories exist
        if os.path.exists("logs/json"):
            print("\nDirectories in logs/json:")
            for item in os.listdir("logs/json"):
                path = os.path.join("logs/json", item)
                if os.path.isdir(path):
                    print(f"  📁 {item}")
                    # Show JSON files in each directory
                    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
                    for json_file in json_files:
                        print(f"    📄 {json_file}")
        else:
            print("logs/json directory does not exist")
        return
    
    print(f"\n✅ Found results for {len(results)} model configurations:")
    for model_name, model_data in results.items():
        print(f"  🔧 {model_name}: {len(model_data)} QPS points")
        for qps in sorted(model_data.keys()):
            print(f"    📊 QPS {qps}")
    
    # Print summary table
    print_summary_table(results)
    
    # Create the plot
    try:
        plot_tpot_vs_qps(results, "test_benchmark_plot.png")
        print("\n✅ Plot created successfully!")
    except Exception as e:
        print(f"\n❌ Error creating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()