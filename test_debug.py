import gc
from test.test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Minimal test
model_path = "../DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load llaisys model
print("Loading llaisys model...")
import traceback
try:
    print("Step 1: Import config...")
    import json
    from pathlib import Path
    model_path_obj = Path(model_path)
    
    print("Step 2: Read config.json...")
    with open(model_path_obj / "config.json") as f:
        config = json.load(f)
    print(f"  Config loaded: nlayer={config['num_hidden_layers']}")
    
    print("Step 3: Creating model...")
    model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
    print("Model loaded successfully!")
except Exception as e:
    print(f"\nError during model loading: {e}")
    traceback.print_exc()
    import sys
    sys.exit(1)

# Simple test
prompt = "Who are you?"
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer.encode(input_content)
print(f"Input tokens: {inputs}")
print(f"Number of input tokens: {len(inputs)}")

print("\nCalling model.generate()...")
try:
    outputs = model.generate(
        inputs,
        max_new_tokens=10,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    print(f"Output tokens: {outputs}")
except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()
