import base64
import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# 🔐 Load OpenAI API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIVLM:
    """Lightweight wrapper around OpenAI's GPT-4o vision model."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 500):
        self.model = model
        self.max_tokens = max_tokens

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _call(self, prompt: str, image: str) -> str:
        base64_img = self._encode_image(image)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def invoke(self, input: Dict[str, str]) -> str:
        prompt = input['prompt']
        image_path = input['image']
        return self._call(prompt, image_path)