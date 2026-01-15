# project/llm/wrapper.py

import os
import yaml
from openai import OpenAI

# Load Config
config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# --- Configuration Setup ---
llm_config = config['llm']
provider = llm_config.get('provider', 'openai').lower()
api_key = os.getenv("OPENAI_API_KEY") or llm_config.get('api_key')

# 1. FIX: Handle Groq/DeepSeek Base URLs automatically
base_url = llm_config.get('base_url')
if not base_url:
    if provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "deepseek":
        base_url = "https://api.deepseek.com/v1"

# 2. FIX: specific key access (config uses 'model', code used 'model_name')
model_name = llm_config.get('model_name') or llm_config.get('model')

# Initialize Client
client = OpenAI(api_key=api_key, base_url=base_url)

def query_llm(system_prompt, user_prompt):
    """
    Wrapper to send a prompt to the configured LLM (Groq/OpenAI).
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error generating response."