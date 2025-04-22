import streamlit as st
import tempfile
import os
import pandas as pd
from scripts.rag_setup.rag import run_custom_rag_pipeline
from scripts.preprocessing.description_gen import generate_descriptions_from_video
from scripts.rag_setup.auto_insert_data import seed_from_csv
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import subprocess


load_dotenv()

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

# Initialize result storage
if "result_top1" not in st.session_state:
    st.session_state["result_top1"] = None
if "result_top3" not in st.session_state:
    st.session_state["result_top3"] = None

st.title("🎥 Interactive Video Learning Platform Through Multimodel RAG")

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


                # Transcribe full audio using OpenAI Whisper
                st.spinner("🗣️ Transcribing full video audio...")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                    # Convert to WAV using ffmpeg
                    subprocess.call([
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                        audio_file.name
                    ])
                    with open(audio_file.name, "rb") as f:
                        resp = client.audio.transcriptions.create(
                            file=f,
                            model="whisper-1"
                        )
                        full_transcript = resp.text.strip()
                        st.session_state["full_transcript"] = full_transcript



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


run_query = st.button("🚀 Naive Approach: Grounding all modalities in text")
if run_query:
    image_path = None
    if image_file:
        st.image(image_file, caption="📸 User Query")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_file.read())
            image_path = tmp_img.name

    result = run_custom_rag_pipeline(user_query=user_query, image_path=image_path, num_of_retrieval=1)
    st.session_state["result_top1"] = result

    st.markdown("## 🎬 Video Clip")
    if "video_file" in st.session_state:
        clip_path = extract_video_clip_with_audio(
            video_path=st.session_state["video_file"],
            #timestamp_ms=result["metadata"]["timestamp_ms"]
            timestamp_ms=result["metadata"][0]["timestamp_ms"]
        )
        st.video(clip_path)
    st.markdown(f"**🧠 LLM Answer:** {result['llm_output']}")
    with st.expander("📦 Retrieved Image Frame + Metadata + Description"):
        st.image(result["image_used"], caption="📸 Matched Frame", width=500)
        st.markdown(f"**📦 Metadata:**")
        st.json(result["metadata"])

run_query_top3 = st.button("🚀 Improved DL approach: attention across all modalities")
if run_query_top3:
    image_path = None
    if image_file:
        st.image(image_file, caption="📸 User Query")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_file.read())
            image_path = tmp_img.name

    #result = run_custom_rag_pipeline(user_query=user_query, image_path=image_path, num_of_retrieval=3)
    result = run_custom_rag_pipeline(
        user_query=user_query,
        image_path=image_path,
        num_of_retrieval=3,
        full_transcript=st.session_state.get("full_transcript", "")
    )
    st.session_state["result_top3"] = result


    # Extract video range based on top 3 metadata timestamps
    timestamps = [meta["timestamp_ms"] for meta in result["metadata"]]
    start_ms = min(timestamps)
    end_ms = max(timestamps)

    st.markdown("## 🎬 Combined Video Clip (Top 3 Context)")
    if "video_file" in st.session_state:
        clip_path = extract_video_clip_with_audio(
            video_path=st.session_state["video_file"],
            timestamp_ms=(start_ms + end_ms) / 2,
            play_before_sec=(end_ms - start_ms) / 2000.0 / 2,  # convert ms to seconds
            play_after_sec=(end_ms - start_ms) / 2000.0 / 2
        )
        st.video(clip_path)
    
    st.markdown(f"**🌠 Gemini 1.5 Pro Answer:** {result['llm_output']}")

    with st.expander("## 📸 Top 3 Retrieved Frames"):
        st.markdown("## 📸 Top 3 Retrieved Frames")
        for i, (desc, img, meta) in enumerate(zip(result["description"], result["image_used"], result["metadata"])):
            st.image(img, caption=f"Frame {i+1} | 🕒 {meta['timestamp_ms']} ms", width=500)
            st.markdown(f"**📝 Description:** {desc}")
            st.markdown("**📦 Metadata:**")
            st.json(meta)
            st.divider()

# --------------- DISPLAY RESULTS --------------- #
with st.expander("## 📝 Recap on these two models performance"):
    # 🔹 Top-1 GPT-4o Result
    if st.session_state["result_top1"]:
        result = st.session_state["result_top1"]
        st.markdown("## 🎬 Video Clip (Top-1 Result)")
        if "video_file" in st.session_state:
            clip_path = extract_video_clip_with_audio(
                video_path=st.session_state["video_file"],
                timestamp_ms=result["metadata"][0]["timestamp_ms"]
            )
            st.video(clip_path)
        st.markdown(f"**🧠 GPT-4o Answer:** {result['llm_output']}")

    # 🔸 Top-3 Gemini Result
    if st.session_state["result_top3"]:
        result = st.session_state["result_top3"]
        timestamps = [meta["timestamp_ms"] for meta in result["metadata"]]
        start_ms = min(timestamps)
        end_ms = max(timestamps)

        st.markdown("## 🎬 Combined Video Clip (Top 3 Result)")
        if "video_file" in st.session_state:
            clip_path = extract_video_clip_with_audio(
                video_path=st.session_state["video_file"],
                timestamp_ms=(start_ms + end_ms) / 2,
                play_before_sec=(end_ms - start_ms) / 2000.0 / 2,
                play_after_sec=(end_ms - start_ms) / 2000.0 / 2
            )
            st.video(clip_path)

        st.markdown(f"**🌠 Gemini 1.5 Pro Answer:** {result['llm_output']}")


# Dev tools
with st.expander("⚙️ Developer Options"):
    if st.button("🗑️ Clear LanceDB Table"):
        clear_lancedb_table()

import shutil
import atexit
def clean_temp_outputs():
    # Remove CSVs
    for csv_file in ["frames_transcripts.csv", "frames_with_descriptions.csv"]:
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"🧹 Removed: {csv_file}")

    # Remove folders
    for folder in ["frames", "audio"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🧹 Removed folder: {folder}")

# 🧹 Register cleanup to run when Streamlit exits
import atexit
atexit.register(clean_temp_outputs)
