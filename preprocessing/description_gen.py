#!/usr/bin/env python3
import os
import cv2
import pandas as pd
import base64
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from openai import OpenAI

from .video_pre import (
    downsample_video,
    detect_scenes,
    extract_key_clips,
    filter_frames,
    frame_index_to_ms,
    grab_frame_at_ms,
)

# -----------------------------------------------------------------------------
# 1) Audio‐segment extraction around a timestamp
# -----------------------------------------------------------------------------
def extract_audio_segment(
    video_path: str,
    timestamp_ms: float,
    before_sec: float = 2.5,
    after_sec: float = 2.5,
    output_audio_path: str = "audio_segment.wav",
) -> str:
    clip = VideoFileClip(video_path)
    t0 = timestamp_ms / 1000.0
    start = max(0.0, t0 - before_sec)
    end   = min(clip.duration, t0 + after_sec)
    sub = clip.subclipped(start, end)
    sub.audio.write_audiofile(output_audio_path, fps=16000, codec="pcm_s16le")
    clip.close()
    return output_audio_path

# -----------------------------------------------------------------------------
# 2) Transcribe via OpenAI v1 client
# -----------------------------------------------------------------------------
def transcribe_with_openai_client(
    client: OpenAI,
    audio_path: str,
    model: str = "whisper-1",
) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=model
        )
    return resp.text.strip()

# -----------------------------------------------------------------------------
# 3) Run the full video→frames→audio→transcript pipeline
# -----------------------------------------------------------------------------
def process_video_to_dataframe(
    video_path: str,
    raw_image_dir: str    = "./frames",
    audio_output_dir: str = "./audio",
    downsample_fps: int   = 3,
    scene_threshold: float= 0.6,
    max_frames: int       = 40,
    before_sec: float     = 2.5,
    after_sec: float      = 2.5,
) -> pd.DataFrame:
    # prepare dirs
    Path(raw_image_dir).mkdir(parents=True, exist_ok=True)
    Path(audio_output_dir).mkdir(parents=True, exist_ok=True)

    # init OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in environment")
    client = OpenAI(api_key=api_key)

    # Stage 1: Downsample for RAG
    frames, fps = downsample_video(video_path, output_fps=downsample_fps)

    # Stage 2: Scene detection
    scenes = detect_scenes(frames, threshold=scene_threshold)

    # Stage 3: Key‐clip extraction
    key_frames = extract_key_clips(frames, scenes)

    # Stage 4: Blur & entropy filtering
    filtered_frames = filter_frames(key_frames, max_frames=max_frames)

    records = []
    print("[Run] Saving raw frames & audio segments...")
    for i, (_frame, original_index) in enumerate(filtered_frames):
        ts_ms = frame_index_to_ms(original_index, fps)

        # — Raw frame from original video —
        raw = grab_frame_at_ms(video_path, ts_ms)
        img_name = f"frame_{i:02d}_{int(ts_ms)}ms.jpg"
        img_path = os.path.join(raw_image_dir, img_name)
        cv2.imwrite(img_path, cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))

        # — Audio segment —
        audio_name = f"audio_{i:02d}_{int(ts_ms)}ms.wav"
        audio_path = os.path.join(audio_output_dir, audio_name)
        extract_audio_segment(
            video_path=video_path,
            timestamp_ms=ts_ms,
            before_sec=before_sec,
            after_sec=after_sec,
            output_audio_path=audio_path,
        )

        # — Transcript —
        transcript = transcribe_with_openai_client(client, audio_path)

        records.append({
            "timestamp_ms": ts_ms,
            "image_path": img_path,
            "audio_path": audio_path,
            "transcript": transcript,
        })
        print(f"[{i:02d}] {int(ts_ms)}ms → “{transcript}”")

    # build DataFrame
    df = pd.DataFrame(records, columns=["timestamp_ms","image_path","audio_path","transcript"])
    return df

# -----------------------------------------------------------------------------
# 4) Use a vision‐enabled LLM to elaborate on each frame+transcript
# -----------------------------------------------------------------------------
def describe_frame_with_transcript(
    image_path: str,
    transcript: str,
    client: OpenAI,
    model: str = "gpt-4o-mini"
) -> str:
    # encode image as base64
    # with open(image_path, "rb") as img_f:
    #     b64 = base64.b64encode(img_f.read()).decode()
    from PIL import Image
    from io import BytesIO
    import base64

    # 1) Load & down‑sample the image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((580, 350), Image.BICUBIC)

    # 2) JPEG‑compress into memory
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=30)

    # 3) Base64‑encode the compressed bytes
    b64 = base64.b64encode(buffer.getvalue()).decode()

    prompt = (
        "Here is the image:\n"
        f"![frame](data:image/jpeg;base64,{b64})\n\n"
        f"Here is the transcript of the image: \"{transcript}\".\n"
        "Using the transcript as context, provide a detailed description capturing ALL visible information. "
        "Include the transcript alongside your description. (response should be under 380 tokens)"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=380
    )
    return resp.choices[0].message.content.strip()

def annotate_dataframe_with_descriptions(
    df: pd.DataFrame,
    image_col: str = "image_path",
    transcript_col: str = "transcript",
    description_col: str = "description"
) -> pd.DataFrame:
    # init client
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    descriptions = []
    for idx, row in df.iterrows():
        desc = describe_frame_with_transcript(
            image_path=row[image_col],
            transcript=row[transcript_col],
            client=client
        )
        descriptions.append(desc)
        print(f"[{idx:02d}] described")
    df[description_col] = descriptions
    return df

def generate_descriptions_from_video(video_file: str) -> pd.DataFrame:
    """
    Full pipeline to process a video → extract frames → transcribe → describe.
    Returns a DataFrame with: timestamp, image_path, audio_path, transcript, description.
    """
    df = process_video_to_dataframe(video_file)
    df.to_csv("frames_transcripts.csv", index=False)

    annotated = annotate_dataframe_with_descriptions(df)
    annotated.to_csv("frames_with_descriptions.csv", index=False)

    return annotated

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    video_file = "mini-ws.mp4"
    final_df = generate_descriptions_from_video(video_file)
    print(final_df.to_string(index=False))

# if __name__ == "__main__":
#     video_file = "mini-ws.mp4"
#     # 1) preprocess frames/audio/transcripts
#     df = process_video_to_dataframe(video_file)
#     # optionally save intermediate CSV
#     df.to_csv("frames_transcripts.csv", index=False)

#     # 2) annotate with LLM descriptions
#     annotated = annotate_dataframe_with_descriptions(df)
#     annotated.to_csv("frames_with_descriptions.csv", index=False)

#     # print final
#     print(annotated.to_string(index=False))