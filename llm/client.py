# project/llm/client.py

import os
import logging
import yaml
from openai import OpenAI

class LLMClient:
    def __init__(self): 
        # Load Config
        config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
        if not os.path.exists(config_path):
             raise FileNotFoundError("config.yaml not found")
             
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.provider = config['llm']['provider']
        self.model = config['llm']['model']
        api_key = config['llm']['api_key']

        if not api_key or api_key == "gsk_...":
             raise ValueError("CRITICAL: Invalid Groq API Key in config.yaml")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

    def generate(self, prompt, system_prompt=None, temperature=0.0):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API Error: {e}")
            return None

    def generate_json(self, prompt, schema_hint):
        """
        Forces JSON output.
        """
        json_prompt = f"{prompt}\n\nIMPORTANT: Output ONLY valid JSON matching this structure: {schema_hint}. Do not wrap in markdown."
        
        response = self.generate(json_prompt, temperature=0.0)
        
        if not response: 
            return "{}"

        clean_response = response
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0]
        elif "```" in clean_response:
            clean_response = clean_response.split("```")[1].split("```")[0]
            
        return clean_response.strip()