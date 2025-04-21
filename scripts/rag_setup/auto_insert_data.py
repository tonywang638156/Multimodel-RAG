#!/usr/bin/env python3
import os
import pandas as pd
import lancedb
from pathlib import Path

from scripts.vectorstores.raw_multimodal_lancedb import RawMultimodalLanceDB
from scripts.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings

def seed_from_csv(
    csv_path: str = "./preprocessing/frames_with_descriptions.csv",
    db_uri: str = "./shared_data/.lancedb",
    table_name: str = "demo_tbl",
):
    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 1a) Normalize paths
    # df["image_path"] = df["image_path"].str.replace(r"^\./frames", "preprocessing/frames", regex=True)
    # df["audio_path"] = df["audio_path"].str.replace(r"^\./audio", "preprocessing/audio", regex=True)
    # ✅ Only use file names relative to current directory
    df["image_path"] = df["image_path"].apply(lambda p: os.path.join("frames", os.path.basename(p)))
    df["audio_path"] = df["audio_path"].apply(lambda p: os.path.join("audio", os.path.basename(p)))


    # 2) Init BridgeTower embedder + LanceDB vectorstore
    embedder = BridgeTowerEmbeddings()
    vectorstore = RawMultimodalLanceDB(
        uri=db_uri,
        embedding_model=embedder,
        table_name=table_name
    )

    # 3) Avoid inserting duplicates
    db = lancedb.connect(db_uri)
    try:
        tbl = db.open_table(table_name)
        existing = {row["metadata"]["image_path"] for _, row in tbl.to_pandas().iterrows()}
    except Exception:
        existing = set()

    # 4) Filter out already-inserted rows
    new = df[~df["image_path"].isin(existing)]
    if new.empty:
        print("🎉 No new frames to seed.")
        return

    # 5) Build the embedding input lists
    texts = new["description"].tolist()
    images = new["image_path"].tolist()
    metadatas = new.apply(lambda row: {
        "image_path": row["image_path"],
        "audio_path": row["audio_path"],
        "description": row["description"],  # ← replaces 'transcript'
        "timestamp_ms": float(row["timestamp_ms"])
    }, axis=1).tolist()

    # 6) Insert into LanceDB
    print(f"📦 Seeding {len(texts)} new multimodal entries into {table_name}…")
    vectorstore.add_text_image_pairs(texts, images, metadatas)
    print("✅ Done.")

if __name__ == "__main__":
    seed_from_csv()
