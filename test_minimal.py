#!/usr/bin/env python3
import sys
import os

print("=== Minimal Qwen2 Test ===\n")

# Step 1: Import llaisys
print("Step 1: Importing llaisys...")
import llaisys
print("  ✓ Success\n")

# Step 2: Load config
print("Step 2: Loading config...")
import json
from pathlib import Path

model_path = Path("../DeepSeek-R1-Distill-Qwen-1.5B")
with open(model_path / "config.json") as f:
    config = json.load(f)
print(f"  ✓ Config loaded: {config['num_hidden_layers']} layers\n")

# Step 3: Create Qwen2 model
print("Step 3: Creating Qwen2 model...")
print(f"  Model path: {model_path}")
print(f"  Device: CPU")

model = llaisys.models.Qwen2(str(model_path), llaisys.DeviceType.CPU)
print("  ✓ Model created\n")

print("=== Test Complete ===")
