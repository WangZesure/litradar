import json
from typing import Dict, Any, Optional
from openai import OpenAI


class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.deepseek.com"),
        )
        self.model = config.get("model", "deepseek-chat")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.3)

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None

    def chat_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        raw = self.chat(system_prompt, user_prompt)
        if not raw:
            return None
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            print(f"[LLM] JSON parse error, raw: {raw[:200]}...")
        return None
