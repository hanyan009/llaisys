import sys
import os
import argparse
from typing import List, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import time
import json
import torch

# Add python path for llaisys
# Adjust this path based on where to_API is relative to python directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../python'))

try:
    import llaisys
    from llaisys.libllaisys import DeviceType
except ImportError:
    print("Error: Could not import llaisys. Make sure you are in the correct directory.")
    sys.exit(1)

# Global variables
model = None
tokenizer = None
server_args = None

app = FastAPI(title="Llaisys API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    stream: Optional[bool] = False

def llaisys_device(name):
    if name == "cpu":
        return DeviceType.CPU
    elif name == "nvidia":
        return DeviceType.NVIDIA
    else:
        raise ValueError(f"Unknown device: {name}")

def load_model():
    global model, tokenizer, server_args
    
    if model is not None:
        return

    # We need to load HF tokenizer for template application and decoding
    from transformers import AutoTokenizer
    
    model_path = server_args.model
    device_name = server_args.device
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Loading llaisys model from {model_path} on {device_name}...")
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    print("Model loaded successfully.")

def stream_generator(input_ids, request):
    generator = model.generate_stream(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=50,
        temperature=request.temperature
    )
    
    generated_ids = []
    previous_text = ""
    
    for token_id in generator:
        generated_ids.append(token_id)
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Calculate new text
        new_text = current_text[len(previous_text):]
        previous_text = current_text
        
        if new_text:
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": new_text
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
    # Final chunk with finish reason
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, tokenizer
    
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation=[{"role": m.role, "content": m.content} for m in request.messages],
        add_generation_prompt=True,
        tokenize=False
    )
    
    input_ids = tokenizer.encode(prompt)
    
    if request.stream:
        return StreamingResponse(
            stream_generator(input_ids, request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=50, 
            temperature=request.temperature
        )
        # generated ids are appended to input_ids (based on qwen2.py implementation)
        # Wait, qwen2.py generate returns full sequence?
        # Yes: output_ids = list(inputs); ... output_ids.append(next_token); return output_ids
        
        generated_ids = output_ids[len(input_ids):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(output_ids)
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--device", type=str, default="nvidia", help="Device to run on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run on")
    
    server_args = parser.parse_args()
    
    # Load model immediately to fail fast
    load_model()
    
    uvicorn.run(app, host=server_args.host, port=server_args.port)
