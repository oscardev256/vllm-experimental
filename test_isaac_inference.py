#!/usr/bin/env python3
"""
Test Isaac model inference with vLLM using the example image.
"""

from vllm import LLM, SamplingParams
from PIL import Image

def main():
    # Initialize the Isaac model with vLLM
    print("Initializing Isaac model with vLLM...")
    llm = LLM(
        model="OscarGD6/Isaac-0.1",
        trust_remote_code=True,
        max_model_len=2048,  # Reduce memory usage
        gpu_memory_utilization=0.7,  # Reduce GPU memory usage
        disable_log_stats=True,
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=200,
        stop_token_ids=[151645, 151643],  # Common stop tokens for chat models
    )
    
    # Load the example image
    image_path = "/home/oscar/dev/vllm/vllm/model_executor/models/example.webp"
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Create the prompt with image
    prompt = "Describe what you see in this image. Look for any people, vehicles, signs, and assess whether it would be safe to cross the street."
    
    # Create multimodal input
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        }
    }
    
    print("Running inference...")
    print(f"Prompt: {prompt}")
    
    # Generate response
    outputs = llm.generate(inputs, sampling_params)
    
    # Print the result
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\nGenerated response:")
        print(f"'{generated_text}'")
        print(f"\nTotal tokens generated: {len(output.outputs[0].token_ids)}")

if __name__ == "__main__":
    main()