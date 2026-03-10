#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python to_API/api_server.py --model ../DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --port 8002
