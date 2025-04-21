from transformers import BridgeTowerProcessor, BridgeTowerModel
from PIL import Image
import torch
from typing import List

# Load once globally
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
model.eval()

# text-image pair embed
def get_dummy_image():
    return Image.new("RGB", (224, 224), color=(255, 255, 255))
def bridge_tower_pair_embed(text: str, image_path: str = None) -> List[float]:
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = get_dummy_image()
        #inputs = processor(text=text, return_tensors="pt")
    inputs = processor(text=text, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.pooler_output[0].tolist()

# text only embed
def bridge_tower_text_only_embed(text: str) -> List[float]:
    # inputs = processor.tokenizer(text, return_tensors="pt")  # ✅ not processor(...)
    # with torch.no_grad():
    #     outputs = model.text_model(**inputs)  # ✅ model.text_model not model.text_encoder
    #     return outputs.last_hidden_state[:, 0, :][0].tolist()  # [CLS] token
    image = get_dummy_image()
    #inputs = processor(text=text, return_tensors="pt")
    inputs = processor(text=text, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.pooler_output[0].tolist()

# image only embed
def bridge_tower_image_only_embed(image_path: str) -> List[float]:
    default_prompt = "Describe the contents of this image and trace it back to its video."
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=default_prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.pooler_output[0].tolist()


# image only embed
# def bridge_tower_image_only_embed(image_path: str) -> List[float]:
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(text="", images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         return outputs.pooler_output[0].tolist()







