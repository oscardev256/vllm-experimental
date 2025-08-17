#!/usr/bin/env python3
"""
Debug script to test AudioPromptCocoDataset loading in isolation
"""
import logging
import sys
import os

# Add vllm to path
sys.path.insert(0, '/home/oscar/dev/vllm')

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_dataset_loading():
    """Test loading the AudioPromptCocoDataset in isolation"""
    logger.info("=== Testing AudioPromptCocoDataset Loading ===")
    
    try:
        # Import after path setup
        from vllm.benchmarks.datasets import AudioPromptCocoDataset
        from transformers import AutoTokenizer
        
        logger.info("Imports successful")
        
        # Create dataset
        logger.info("Creating AudioPromptCocoDataset...")
        dataset = AudioPromptCocoDataset(
            dataset_path="OscarGD6/audio-prompt-coco-balanced-subset",
            dataset_subset=None,
            dataset_split="train",
            random_seed=0,
            no_stream=False,
        )
        logger.info("Dataset created successfully")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        logger.info("Tokenizer loaded successfully")
        
        # Try to sample just 1 request
        logger.info("Sampling 1 request...")
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=1,
            output_len=64,
        )
        
        logger.info(f"SUCCESS: Got {len(requests)} requests")
        if requests:
            req = requests[0]
            logger.info(f"Sample request - prompt_len: {req.prompt_len}, output_len: {req.expected_output_len}")
            logger.info(f"Has multimodal data: {req.multi_modal_data is not None}")
            if req.multi_modal_data:
                logger.info(f"Multimodal keys: {list(req.multi_modal_data.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_args():
    """Test if the issue is in benchmark argument parsing"""
    logger.info("=== Testing Benchmark Argument Parsing ===")
    
    try:
        from vllm.benchmarks.datasets import get_samples
        from transformers import AutoTokenizer
        import argparse
        
        logger.info("Creating mock args...")
        args = argparse.Namespace()
        args.dataset_name = "hf"
        args.dataset_path = "OscarGD6/audio-prompt-coco-balanced-subset"
        args.hf_subset = None
        args.hf_split = "train"
        args.hf_output_len = 64
        args.num_prompts = 1
        args.seed = 0
        args.no_stream = False
        args.endpoint_type = "openai-audio"  # Set to audio endpoint
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        
        logger.info("Calling get_samples...")
        requests = get_samples(args, tokenizer)
        
        logger.info(f"SUCCESS: Got {len(requests)} requests via get_samples")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting dataset debug tests...")
    
    # Test 1: Direct dataset loading
    success1 = test_dataset_loading()
    
    # Test 2: Via benchmark function
    success2 = test_benchmark_args()
    
    if success1 and success2:
        logger.info("✅ All tests passed - dataset loading works!")
    else:
        logger.error("❌ Some tests failed - check logs above")
        sys.exit(1)