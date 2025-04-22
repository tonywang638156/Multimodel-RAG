import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiVLM:
    def __init__(self, model="gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model)

    def invoke(self, input):
        prompt = input["prompt"]
        image_paths = input["image"]
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        response = self.model.generate_content(
            [prompt] + images,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=800
            )
        )
        return response.text