import requests
import json
import sys

url = "http://localhost:8002/v1/chat/completions"
data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "messages": [{"role": "user", "content": "请用中文介绍一下你自己。"}],
    "stream": True,
    "max_tokens": 128
}

print(f"Sending request to {url}...")
try:
    with requests.post(url, json=data, stream=True) as r:
        if r.status_code != 200:
            print(f"Error: {r.status_code} - {r.text}")
            sys.exit(1)
            
        print("Response:")
        for line in r.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        json_data = json.loads(line[6:])
                        content = json_data['choices'][0]['delta'].get('content', '')
                        print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        pass
    print("\n\nTest completed successfully.")
except Exception as e:
    print(f"\nConnection error: {e}")
    print("Make sure the server is running.")
