import streamlit as st
import tempfile
import os
import pandas as pd
from rag_setup.rag import run_custom_rag_pipeline
from preprocessing.description_gen import generate_descriptions_from_video
from rag_setup.auto_insert_data import seed_from_csv
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip

def clear_lancedb_table(db_uri: str = "./shared_data/.lancedb", table_name: str = "demo_tbl"):
    import lancedb
    db = lancedb.connect(db_uri)
    db.drop_table(table_name, ignore_missing=True)
    st.success(f"🗑️ Cleared `{table_name}` from LanceDB.")
    st.session_state["processed"] = False

# ✅ Video clip extractor
def extract_video_clip_with_audio(
    video_path: str,
    timestamp_ms: float,
    output_video_path: str = "./shared_data/splitted_videos",
    output_video_name: str = "video_tmp.mp4",
    play_before_sec: float = 2.5,
    play_after_sec: float = 2.5,
    width_factor: float = 1,
    height_factor: float = 1
) -> str:
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_video_path, output_video_name)

    clip = VideoFileClip(video_path)
    t0 = timestamp_ms / 1000.0
    start = max(0.0, t0 - play_before_sec)
    end = min(clip.duration, t0 + play_after_sec)

    sub = clip.subclipped(start, end)
    if clip.audio:
        audio_sub = clip.audio.subclipped(start, end)
        sub = sub.with_audio(audio_sub)

    new_w = int(sub.w * width_factor)
    new_h = int(sub.h * height_factor)
    sub = sub.resized((new_w, new_h))

    sub.write_videofile(
        output_path,
        codec="libx264",
        audio=True,
        audio_codec="aac",
        audio_fps=44100,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=4,
        logger=None,
    )
    clip.close()
    return output_path

# --- UI setup ---
st.set_page_config(page_title="Video ↦ Semantic Frames")
st.title("🎥 Upload a Video and Get Descriptive Semantic Frames")

uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if uploaded_file:
    # ✅ Read video bytes for native playback with sound
    video_bytes = uploaded_file.read()
    st.video(video_bytes)

    if "processed" not in st.session_state or not st.session_state["processed"]:
        with st.spinner("🔍 Processing video into smart frames..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                video_path = tmp.name
                st.session_state["video_file"] = video_path  # ✅ Save for RAG video clip

            df: pd.DataFrame = generate_descriptions_from_video(video_path)
            df.to_csv("frames_with_descriptions.csv", index=False)
            #os.remove(video_path)

        with st.spinner("📦 Inserting data into LanceDB..."):
            seed_from_csv(
                csv_path="./frames_with_descriptions.csv",
                db_uri="./shared_data/.lancedb",
                table_name="demo_tbl"
            )
            st.session_state["processed"] = True

        st.success("✅ Done! Showing descriptive results...")

        # ✅ Optional preview
        with st.expander("👀 Do you want to see the first 3 semantic frame examples?"):
            for i, row in df.head(3).iterrows():
                st.image(row["image_path"], caption=f"🕒 {row['timestamp_ms']:.0f} ms", width=500)
                st.markdown(f"**Transcript:** {row['transcript']}")
                st.markdown(f"**Description:** {row['description']}")
                st.divider()

        # CSV download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, "frames_with_descriptions.csv", "text/csv")

    else:
        st.info("✅ Video already processed. Clear LanceDB to reprocess.")

# -------------------- RAG QUERY INTERFACE -------------------- #
st.subheader("🔍 Ask a Question (Text, Image, or Both)")

user_query = st.text_input("📝 Enter your question (optional)")
image_file = st.file_uploader("🖼️ Upload an image (optional)", type=["jpg", "jpeg", "png"], key="query_img")


run_query = st.button("🚀 Run RAG Query")
if run_query:
    image_path = None
    if image_file:
        st.image(image_file, caption="📸 User Query")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_file.read())
            image_path = tmp_img.name

    result = run_custom_rag_pipeline(user_query=user_query, image_path=image_path)
    st.markdown("## 🎬 Video Clip")
    if "video_file" in st.session_state:
        clip_path = extract_video_clip_with_audio(
            video_path=st.session_state["video_file"],
            timestamp_ms=result["metadata"]["timestamp_ms"]
        )
        st.video(clip_path)
    st.markdown(f"**🧠 LLM Answer:** {result['llm_output']}")
    with st.expander("📦 Retrieved Image Frame + Metadata + Description"):
        st.image(result["image_used"], caption="📸 Matched Frame", width=500)
        st.markdown(f"**📦 Metadata:**")
        st.json(result["metadata"])

# Dev tools
with st.expander("⚙️ Developer Options"):
    if st.button("🗑️ Clear LanceDB Table"):
        clear_lancedb_table()
