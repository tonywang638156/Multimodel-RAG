from typing import List, Optional
from tqdm import tqdm
#from .local_bridgetower_utils import bridge_tower_local_embed  # Local BridgeTower wrapper
from .local_bridgetower_utils import (
    bridge_tower_pair_embed,
    bridge_tower_text_only_embed,
    bridge_tower_image_only_embed,
)


class BridgeTowerEmbeddings:
    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts_only([text])[0]

    def embed_texts_only(self, texts: List[str]) -> List[List[float]]:
        return [bridge_tower_text_only_embed(text) for text in texts]

    def embed_images_only(self, image_paths: List[str]) -> List[List[float]]:
        return [bridge_tower_image_only_embed(img_path) for img_path in image_paths]

    def embed_image_text_pairs(self, texts: List[str], images: List[str]) -> List[List[float]]:
        assert len(texts) == len(images), "texts and images must match"
        return [bridge_tower_pair_embed(t, i) for t, i in zip(texts, images)]
    
    def embedding_dimension(self) -> int:
        return 1536  # BridgeTower-base returns pooler_output of shape [1, 1536]
    
    def embed_user_query(self, query_text: Optional[str], image_path: Optional[str]) -> List[float]:
        if query_text and image_path:
            return self.embed_image_text_pairs([query_text], [image_path])[0]
        elif image_path:
            return bridge_tower_image_only_embed(image_path)
        elif query_text:
            return bridge_tower_text_only_embed(query_text)
        else:
            raise ValueError("Must provide at least a query text or an image.")

