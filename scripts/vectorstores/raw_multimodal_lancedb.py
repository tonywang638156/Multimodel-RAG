import uuid
from typing import List, Optional
import lancedb
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RawMultimodalLanceDB:
    def __init__(
        self,
        uri: str,
        embedding_model,
        table_name: str = "vectorstore",
    ):
        self.embedding_model = embedding_model
        self.db = lancedb.connect(uri)
        self.table_name = table_name

        embedding_dim = self.embedding_model.embedding_dimension()
        if table_name not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("text", pa.string()),
                ("image_path", pa.string()),
                ("vector", pa.list_(pa.float32(), embedding_dim)),
                ("metadata", pa.struct([
                    ("image_path", pa.string()),
                    ("audio_path", pa.string()),
                    ("description", pa.string()),  # 🟢 updated
                    ("timestamp_ms", pa.float32())
                ]))
            ])
            self.db.create_table(table_name, schema=schema)
        self.table = self.db.open_table(table_name)

    def add_text_image_pairs(
        self,
        texts: List[str],
        image_paths: List[str],
        metadatas: Optional[List[dict]] = None,
    ):
        assert len(texts) == len(image_paths) == len(metadatas), "Length mismatch."

        existing_paths = set(self.table.to_arrow()["image_path"].to_pylist())
        docs = []

        for i in range(len(texts)):
            path = image_paths[i]
            if path in existing_paths:
                print(f"⚠️ Skipping duplicate image: {path}")
                continue

            uid = str(uuid.uuid5(uuid.NAMESPACE_URL, texts[i] + path))
            embedding = self.embedding_model.embed_image_text_pairs([texts[i]], [path])[0]

            doc = {
                "id": uid,
                "text": texts[i],
                "image_path": path,
                "vector": embedding,
                "metadata": metadatas[i],
            }
            docs.append(doc)

        if docs:
            self.table.add(docs)
            print(f"✅ Inserted {len(docs)} new records.")
        else:
            print("🟡 No new records inserted.")

    def similarity_search(self, query_text: Optional[str] = None, image_path: Optional[str] = None, k: int = 1):
        if not query_text and not image_path:
            raise ValueError("You must provide at least text or image for query.")

        if query_text and image_path:
            query_emb = self.embedding_model.embed_image_text_pairs([query_text], [image_path])[0]
        elif image_path:
            #query_emb = self.embedding_model.embed_images_only(image_path)
            query_emb = self.embedding_model.embed_images_only([image_path])[0]
        elif query_text:
            #query_emb = self.embedding_model.embed_texts_only(query_text, None)
            query_emb = self.embedding_model.embed_texts_only([query_text])[0]


        results = self.table.search(query_emb, vector_column_name="vector").limit(k).to_list()

        print("\n📈 Cosine Similarities:")
        all_vectors = np.array(self.table.to_arrow()["vector"].to_pylist())
        query_vector = np.array(query_emb).reshape(1, -1)
        similarities = cosine_similarity(query_vector, all_vectors)[0]

        for i, score in enumerate(similarities):
            meta = self.table.to_arrow()["metadata"][i].as_py()
            print(f"{i+1}. 📸 {meta['image_path']} → 🎯 Similarity: {score:.4f}")
        return results
