#!/usr/bin/env python3
"""
Run benchmark comparison across multiple model configurations and QPS rates.
Similar to the experiment in Figure 3 of the RedHat article.
"""

import os
import time
import subprocess

# Configuration
TOTAL_SECONDS = 60  # Longer test for stable results
QPS_RATES = [1, 2, 4, 6, 8, 10, 12, 14, 16]  # Range similar to Figure 3

# Model configurations to compare
MODEL_CONFIGS = [
    {
        "name": "fp16", 
        "model": "your-model-path-fp16",
        "extra_args": ""
    },
    {
        "name": "w8a8", 
        "model": "your-model-path-w8a8",
        "extra_args": "--quantization awq"  # Adjust based on your quantization method
    },
    {
        "name": "w4a16", 
        "model": "your-model-path-w4a16", 
        "extra_args": "--quantization gptq"  # Adjust based on your quantization method
    },
]

def run_benchmark(model_config, qps, num_prompts, result_dir, log_file):
    """Run a single benchmark configuration."""
    
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--endpoint-type", "openai-chat", 
        "--model", model_config["model"],
        "--endpoint", "/v1/chat/completions",
        "--dataset-name", "hf",
        "--dataset-path", "OscarGD6/audio-prompt-coco-balanced-subset",
        "--hf-split", "train",
        "--hf-output-len", "64",
        "--num-prompts", str(num_prompts),
        "--request-rate", str(qps),
        "--max-concurrency", "10",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,95,99",
        "--save-result",
        "--save-detailed", 
        "--result-dir", result_dir,
        "--ready-check-timeout-sec", "3",
        "--enable-multimodal-chat"  # Add this for your multimodal dataset
    ]
    
    # Add any extra arguments for this model config
    if model_config["extra_args"]:
        cmd.extend(model_config["extra_args"].split())
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the benchmark and save output
    with open(log_file, 'w') as f:
        try:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, 
                                  text=True, timeout=600)  # 10 minute timeout
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Benchmark timed out for {model_config['name']} QPS {qps}")
            return False
        except Exception as e:
            print(f"Error running benchmark: {e}")
            return False

def main():
    """Run the full benchmark comparison."""
    
    # Create output directories
    os.makedirs("logs/text", exist_ok=True)
    os.makedirs("logs/json", exist_ok=True)
    
    ts = int(time.time())
    
    total_runs = len(MODEL_CONFIGS) * len(QPS_RATES)
    current_run = 0
    
    print(f"Starting benchmark comparison with {len(MODEL_CONFIGS)} models and {len(QPS_RATES)} QPS rates")
    print(f"Total runs: {total_runs}")
    print("=" * 60)
    
    for model_config in MODEL_CONFIGS:
        print(f"\n🚀 Testing model: {model_config['name']}")
        
        for qps in QPS_RATES:
            current_run += 1
            num_prompts = TOTAL_SECONDS * qps
            
            print(f"\n  📊 Run {current_run}/{total_runs}: {model_config['name']} at {qps} QPS ({num_prompts} prompts)")
            
            # Setup paths using the same format as your current structure
            log_file = f"logs/text/{model_config['name']}_{qps}.log"
            result_dir = f"logs/json/{model_config['name']}_{qps}_{ts}"
            os.makedirs(result_dir, exist_ok=True)
            
            # Run the benchmark
            success = run_benchmark(model_config, qps, num_prompts, result_dir, log_file)
            
            if success:
                print(f"  ✅ Completed successfully")
            else:
                print(f"  ❌ Failed - check {log_file}")
            
            # Small delay between runs
            time.sleep(5)
    
    print("\n" + "=" * 60)
    print("🎉 Benchmark comparison completed!")
    print("\nTo plot the results, run:")
    print("  python plot_benchmark_results.py")

if __name__ == "__main__":
    main()