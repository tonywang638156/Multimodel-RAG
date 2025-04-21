import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
from scipy.stats import entropy
import matplotlib.pyplot as plt


# ----------------------------- 1. Downsample Video ----------------------------- #
def downsample_video(video_path, output_fps=3, resize_to=(224, 224)):
    cap = cv2.VideoCapture(video_path)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(orig_fps // output_fps)

    print(f"[INFO] Original FPS: {orig_fps}")
    print(f"[INFO] Total original frames: {total_frames}")
    print(f"[INFO] Frame interval for downsampling: {frame_interval}")

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, resize_to)
            frames.append((frame_rgb, frame_id))  # ⬅️ save as (image, original_index)
        frame_id += 1

    cap.release()

    print(f"[INFO] Total frames after downsampling: {len(frames)}")
    return frames, orig_fps


# -------------------------- 2. Scene Detection (histogram diff) -------------------------- #
def detect_scenes(frames, threshold=0.6):
    scenes = []
    start = 0
    for i in range(1, len(frames)):
        img1 = frames[i-1][0]
        img2 = frames[i][0]
        hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if diff < threshold:
            scenes.append((start, i))
            start = i
    scenes.append((start, len(frames)-1))
    return scenes

# ------------------- 3. Key Clip Detection (SSIM peaks & valleys) ------------------- #
def extract_key_clips(frames, scenes):
    key_frames = []
    for (start, end) in scenes:
        scene = frames[start:end+1]
        similarities = []
        for i in range(1, len(scene)):
            sim = ssim(scene[i-1][0], scene[i][0], win_size=7, channel_axis=-1)
            similarities.append(sim)

        if not similarities:
            continue

        sims = np.array(similarities)
        mean = sims.mean()
        std = sims.std()
        for i, sim in enumerate(sims):
            if sim < (mean - std):
                key_frames.append(scene[i])  # (frame, index)
                break
        key_frames.append(scene[-1])
    return key_frames

# -------------------- 4. Filter: Blur Detection + Entropy Ranking -------------------- #
def is_blurry(image, threshold=1000.0):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold

def compute_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = hist.ravel() / hist.sum()
    return entropy(hist)

def filter_frames(frames, max_frames=40):
    filtered = []
    scored = []

    for (frame, idx) in frames:
        if not is_blurry(frame):
            score = compute_entropy(frame)
            scored.append((score, frame, idx))

    scored.sort(reverse=True)  # High-entropy first
    for i in range(min(max_frames, len(scored))):
        _, frame, idx = scored[i]
        filtered.append((frame, idx))
    return filtered

def frame_index_to_ms(index, fps):
    return (index / fps) * 1000.0

# ------------------------------- Full Pipeline ------------------------------- #
def preprocess_video(video_path, output_dir="selected_frames"):
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Downsampling...")
    downsampled = downsample_video(video_path)

    print("🎬 Detecting scenes...")
    scenes = detect_scenes(downsampled)

    print("🧩 Extracting key clips...")
    key_frames = extract_key_clips(downsampled, scenes)

    print("🧹 Filtering frames...")
    selected_frames = filter_frames(key_frames)

    print(f"✅ Selected {len(selected_frames)} high-information frames.")

    for idx, frame in enumerate(selected_frames):
        path = os.path.join(output_dir, f"frame_{idx:02d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return selected_frames

def frame_index_to_ms(index, fps):
    return (index / fps) * 1000.0

def grab_frame_at_ms(video_path: str, timestamp_ms: float):
    """
    Seek to `timestamp_ms` in the video and grab exactly that frame.
    Returns the frame as an RGB numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    # Seek to the desired time (milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame at {timestamp_ms} ms")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    videoPath = "./ws.mp4"
    print("📥 Step 1: Downsampling video...")
    frames, fps = downsample_video(videoPath)
    print(f"✅ Original frames (approx): {(fps // 3) * len(frames)} (based on downsampling rate)")
    print(f"✅ Downsampled frames: {len(frames)}")

    print("\n🎬 Step 2: Detecting scenes...")
    scenes = detect_scenes(frames)
    print(f"✅ Scenes detected: {len(scenes)}")
    for i, (start, end) in enumerate(scenes[:5]):
        print(f"Scene {i}: from frame {start} to {end}")

    print("\n🧩 Step 3: Extracting key frames...")
    key_frames = extract_key_clips(frames, scenes)
    print(f"✅ Key frames extracted: {len(key_frames)}")
    # Visualize key frames
    print("\n🖼️ Showing extracted key frames...")
    for i, (img, idx) in enumerate(key_frames):
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Key Frame {i} - Index {idx}")
        plt.axis("off")
        plt.show()

    print("\n🧹 Step 4: Filtering high-quality frames (blur + entropy)...")
    filtered_frames = filter_frames(key_frames, max_frames=40)
    print(f"✅ Final filtered frames: {len(filtered_frames)}")

    print("\n🕒 Step 5: Timestamps for Filtered Frames")
    for i, (frame, original_index) in enumerate(filtered_frames):
        timestamp_ms = frame_index_to_ms(original_index, fps)
        print(f"[Filtered Frame {i:02d}] Index = {original_index}, Timestamp = {timestamp_ms:.2f} ms")

    # Optional: visually compare key frames vs. filtered ones
    VIDEO_PATH = videoPath  # or whatever your source video is
    print("\n🔍 Verifying filtered frames against the original video:")
    for i, (filtered_img, original_index) in enumerate(filtered_frames):
        ts_ms = frame_index_to_ms(original_index, fps)
        print(f"\n[Frame {i:02d}] index={original_index}, timestamp={ts_ms:.2f} ms")

        # 1) Show your filtered frame
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(filtered_img)
        plt.title(f"Filtered Frame {i}")
        plt.axis("off")

        # 2) Show the key-frame you selected (optional)
        kf = next(f for f in key_frames if f[1] == original_index)
        plt.subplot(1, 3, 2)
        plt.imshow(kf[0])
        plt.title(f"Key Frame {i}")
        plt.axis("off")

        # 3) Grab the raw frame from the original video at that ms
        raw = grab_frame_at_ms(VIDEO_PATH, ts_ms)
        plt.subplot(1, 3, 3)
        plt.imshow(raw)
        plt.title(f"Raw @ {int(ts_ms)}ms")
        plt.axis("off")

        plt.show()
