import gc
import time
import torch
#import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration#, BitsAndBytesConfig

import sys
sys.path.append("home/oscar/dev/vllm")
from vllm import LLM, EngineArgs, SamplingParams
import logging

logging.getLogger().setLevel(logging.ERROR)

# -------------------------
# Memory Cleanup Function
# -------------------------
def clear_memory():
    if 'llm' in globals(): del globals()['llm']
    if 'model' in globals(): del globals()['model']
    time.sleep(2)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# -------------------------
# Dataset Preparation
# -------------------------
train_idx_end = 40
eval_idx_end = 80

#ds = load_dataset("OscarGD6/audio-command-dev", split="train")

# Load both datasets
ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train")
#ds2 = load_dataset("OscarGD6/audio-command-dev", split="train")

max_pixels=1024 * 28 * 28
#ds_filtered = ds.select(idx for idx in range(len(ds)//4) if (len(ds[idx]["objects"]["bbox"]) == 1) and (ds[idx]["width"]*ds[idx]["height"]) <= max_pixels)
ds_filtered = ds.select(idx for idx in range(8) if (len(ds[idx]["objects"]["bbox"]) == 1) and (ds[idx]["width"]*ds[idx]["height"]) <= max_pixels)
#common_features = ["image_id", "image", "width", "height", "objects"]
#ds_filtered = ds.remove_columns([col for col in ds.column_names if col not in common_features])
print(f"# of filtered images = {len(ds_filtered)}")
#coco_eval = ds_filtered.select(range(128))
#coco_eval = ds_filtered.select(range(61))

# -------------------------
# Common Setup
# -------------------------
processor = Qwen2VLProcessor.from_pretrained(
    "OscarGD6/qwen2vl-nutrition-label-detection-merged-weights",
    use_fast=True,
    #max_pixels=1024 * 28 * 28,
    max_pixels=128 * 28 * 28,
    #size={'shortest_edge': 3136, 'longest_edge': 1024 * 28 * 28},
    size={'shortest_edge': 3136, 'longest_edge': 128 * 28 * 28},
)

# -------------------------
# vLLM Setup
# -------------------------
engine_args = EngineArgs(
    #model="OscarGD6/qwen2vl-nutrition-label-detection-merged-weights",
    #model="Qwen/Qwen2-VL-2B-Instruct",
    model="OscarGD6/Isaac-0.1",
    max_model_len=256,
    max_num_batched_tokens=256,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={
        "max_pixels": 128 * 28 * 28,  # or your target pixel count
    },
    #dtype="bfloat16",
    gpu_memory_utilization=0.82,  # ← reduce from 0.9 to 0.7
    #cpu_offload_gb=10,              # Reserve 10 GB on CPU
    quantization="bitsandbytes",
    enforce_eager=True,
    trust_remote_code=True
)
llm = LLM(**engine_args.__dict__)
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, skip_special_tokens=True)

# -------------------------
# Benchmark Functions
# -------------------------
def vllm_eval(batch_size, eval_subset):
    batch_latencies = []
    for i in range(0, len(eval_subset), batch_size):
        batch = eval_subset.select(range(i, min(i + batch_size, len(eval_subset))))
        prompts = []
        for sample in batch:
            img = sample["image"].convert("RGB")
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Detect the bounding box of the nutrition table."}
                ]
            }]
            inputs = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False, return_dict=True
            )
            prompts.append({"prompt": inputs, "multi_modal_data": {"image": img}})

        start = time.time()
        _ = llm.generate(prompts, sampling_params)
        batch_time = time.time() - start
        batch_latencies.append(batch_time)

    avg_latency = sum(batch_latencies) / len(batch_latencies)
    return avg_latency

def hf_eval(batch_size, eval_subset):
    batch_latencies = []
    for i in range(0, len(eval_subset), batch_size):
        batch = eval_subset.select(range(i, min(i + batch_size, len(eval_subset))))
        conversation = []
        for sample in batch:
            img = sample["image"].convert("RGB")
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Detect the bounding box of the nutrition table."}
                ]
            })

        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=128)
        torch.cuda.synchronize()
        batch_time = time.time() - start
        batch_latencies.append(batch_time)

    avg_latency = sum(batch_latencies) / len(batch_latencies)
    return avg_latency

print(vllm_eval(1, ds_filtered.select(range(1))))