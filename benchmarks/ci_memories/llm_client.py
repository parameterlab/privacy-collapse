# llm_client.py
import os
import json
import asyncio
from typing import Any, Dict, Optional
from litellm import completion, acompletion
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model_name: str, temperature: float = 0.0, local: bool = False, role: str = "target", base_model=None):
        self.model_name = model_name
        self.temperature = temperature
        self.gen_args = {}
        self.gen_args["model"] = model_name
        self.gen_args["temperature"] = temperature
        # self.gen_args["api_key"] = os.getenv("OPENAI_API_KEY")

        if (role == "target") and (local is True):
            self.gen_args["model"] = f"hosted_vllm/{model_name}"
            self.gen_args["base_url"] = "http://0.0.0.0:8000/v1"
            self.gen_args["max_tokens"] = 1024

        if base_model and (role == 'target'):
            self.gen_args["base_model"] = base_model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await acompletion(messages=messages, **self.gen_args)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return ""

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Forces JSON output mode or strips markdown."""
        content = await self.generate(prompt)
        try:
            # Clean up markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"JSON Decode Error for {self.model_name}. Content: {content[:100]}...")
            return {}