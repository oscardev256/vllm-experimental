import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

#from vllm import LLM
def show_type(model): 
    print(type(model))

import sys
if __name__ == '__main__':
  #sys.path.append("/content/vllm")
  #model="Qwen/Qwen2-VL-2B-Instruct"
  #model="Qwen/Qwen3-VL-30B-A3B-Instruct"
  #model="PerceptronAI/Isaac-0.1"
  #model="OscarGD6/Isaac-0.1"
  #model="Qwen/Qwen2-VL-2B-Instruct"
  from vllm import LLM, EngineArgs, SamplingParams
  
  # -------------------------
  # vLLM Setup
  # -------------------------
  engine_args = EngineArgs(
      #model="OscarGD6/qwen2vl-nutrition-label-detection-merged-weights",
      #model="Qwen/Qwen2-VL-2B-Instruct",
      model="OscarGD6/Isaac-0.1",
      max_model_len=256,#4992,  # Use the suggested max length from error message
      max_num_batched_tokens=256,
      max_num_seqs=1,
      limit_mm_per_prompt={"image": 1},
      #mm_processor_kwargs={
      #    "max_pixels": 128 * 28 * 28,  # or your target pixel count
      #},
      #dtype="bfloat16",
      gpu_memory_utilization=0.82,  # ← reduce from 0.9 to 0.7
      #cpu_offload_gb=10,              # Reserve 10 GB on CPU
      #quantization="bitsandbytes",
      enforce_eager=True,
      trust_remote_code=True
  )
  llm = LLM(**engine_args.__dict__)

  #llm = LLM(model=model, trust_remote_code=True, quantization="bitsandbytes")  # Name or path of your model
  #llm = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=0.7)
  #llm = LLM(model=model)  # Name or path of your model
  #llm.apply_model(lambda model: print(type(model)))
  #llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")
  #llm.apply_model(show_type)
  print("Model loaded successfully! Now testing with image inference...")
  
  # Test with a simple text prompt first
  from vllm import SamplingParams
  text_prompt = "Hello, how are you?"
  text_out = llm.generate([text_prompt], SamplingParams(max_tokens=20, temperature=0.0))
  print(f"Text-only test: '{text_out[0].outputs[0].text}'")
  
  # Test with image using proper Isaac format
  from PIL import Image
  image_path = "/home/oscar/dev/vllm/vllm/model_executor/models/example.webp"
  image = Image.open(image_path).convert("RGB")
  
  # Load the Isaac processor using our local implementation
  import sys
  sys.path.insert(0, "/home/oscar/dev/vllm/vllm/model_executor/models")
  from isaac import IsaacProcessor
  from transformers import AutoTokenizer
  
  #from modular_isaac import IsaacProcessor as OfficialIsaacProcessor
  hf_repo = "PerceptronAI/Isaac-0.1"
  

  # Load tokenizer from HuggingFace and create our processor
  tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
  processor = IsaacProcessor(tokenizer=tokenizer)
  
  # Create conversation in Isaac's expected format
  conversation2 = [{
      "role": "user", 
      "content": [
          {"type": "text", "text": "<hint>BOX</hint>"},
          {"type": "image", "image": image},
          {"type": "text", "text": "Determine whether it is safe to cross the street. Look for signage and moving traffic."}
      ]
  }]
  
  def document_to_messages(
      document: list[dict], vision_token: str = "<|image_pad|>"
  ) -> tuple[list[dict[str, str]], list]:
      """
      Convert a Document to messages format compatible with chat templates.
      Each content turn creates its own message entry.
      """
      messages = []
      images = []

      for item in document:
          itype = item.get("type")
          if itype == "text":
              content = item.get("content")
              if content:
                  messages.append(
                      {
                          "role": item.get("role", "user"),
                          "content": content,
                      }
                  )
          elif itype == "image":
              img = item.get("image")  # Use "image" key instead of "content"
              if img:
                  images.append(img)
                  messages.append(
                      {
                          "role": item.get("role", "user"),
                          "content": vision_token,
                      }
                  )

      return messages, images

  # Create a dummy multimodal input (text + image) without external schema dependencies
  # Each item is a dict with keys: type in {"text","image"}, content (string), role (optional)
  conversation = [
      {
          "type": "text",
          "content": "<hint>BOX</hint>",
          "role": "user",
      },
      {
          "type": "image",
          "image": image,
          "role": "user",
      },
      {
          "type": "text",
          "content": "Determine whether it is safe to cross the street. Look for signage and moving traffic.",
          "role": "user",
      },
  ]
  
  # Convert to messages format
  messages, images = document_to_messages(conversation)
  
  # Apply chat template to get properly formatted input
  formatted_inputs = processor.apply_chat_template(
      messages, add_generation_prompt=True, tokenize=False
  )
  
  print(f"DEBUG: Formatted inputs = {formatted_inputs}")
  
  # Create multimodal input for vLLM
  inputs = {
      "prompt": formatted_inputs,
      "multi_modal_data": {
          "image": image
      }
  }
  
  # Use Isaac model's recommended sampling parameters
  sampling_params = SamplingParams(
      temperature=0.0,        # Slightly higher than 0 for some creativity
      #temperature=0.01,        # Slightly higher than 0 for some creativity
      #top_p=0.001,             # Nucleus sampling
      #top_k=1,
      max_tokens=512,        # Longer responses
      #repetition_penalty=1.0,  # Reduce repetition
      #stop=["<|im_end|>", "<|endoftext|>"]
  )
  #outputs = llm.generate(inputs, sampling_params)
  #sampling_params = SamplingParams(temperature=0.0, max_tokens=500)
  #sampling_params = SamplingParams(temperature=0.0)
  outputs = llm.generate(inputs, sampling_params)
  
  print(f"Image inference result: '{outputs[0].outputs[0].text}'")
  print(f"Full inference result: '{outputs}'")
  decoded_text = tokenizer.decode(outputs[0].outputs[0].token_ids)
  print(f"Decoded text: {decoded_text}")
  print("End of process...")