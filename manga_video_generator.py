import os
import cv2
import numpy as np
import random
import json
from moviepy import *
from ultralytics import YOLO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import multiprocessing

# YOLO cache for repeat runs
CACHE_FILE = "yolo_cache.json"
try:
    yolo_cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
except:
    yolo_cache = {}
# =========================
# ⚙️ SETTINGS
# =========================
VIDEO_SIZE = (1080, 1920)          # Portrait HD
FPS = 24                           # Cinematic feel
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLOv8 model for focus detection
# You can replace yolov8n.pt with any anime/face model if you have it
model = None

# =========================
# 🎞️ Helper Functions
# =========================

def fast_blur(img):
    """Fast blur background using downscale + blur + upscale."""
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    blur = cv2.blur(small, (15, 15))
    return cv2.resize(blur, VIDEO_SIZE)

def smooth_zoom_pan(t, duration, start_zoom, end_zoom, start_pos, end_pos):
    """Smooth easing zoom & pan (Ken Burns effect) with cubic easing."""
    progress = t / duration
    progress = progress**2 * (3 - 2 * progress)  # cubic easing for natural accel/decel
    zoom = start_zoom + (end_zoom - start_zoom) * progress
    x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
    return zoom, (x, y)

def handheld_motion(t):
    """Subtle jitter for hand-held realism."""
    return (2.5 * np.sin(2 * np.pi * t * 0.5), 1.5 * np.cos(2 * np.pi * t * 0.6))

def create_cinematic_clip(img_path, duration):
    """Creates a cinematic clip from one image with zoom/pan and blurred background."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Failed to load image {img_path}, skipping")
        return None
    h, w, _ = img.shape

    # --- YOLO detection with per-image caching ---
    cache_path = img_path + ".yolo.json"
    if os.path.exists(cache_path):
        try:
            boxes = np.array(json.load(open(cache_path)))
        except:
            boxes = []
            results = model.predict(img, device='mps', half=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []
    else:
        results = model.predict(img, device='mps', half=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []
        with open(cache_path, "w") as f:
            json.dump(boxes.tolist() if hasattr(boxes, 'tolist') else boxes, f)

    # Default to center crop if no detection
    if len(boxes) == 0:
        x1, y1, x2, y2 = w * 0.25, h * 0.25, w * 0.75, h * 0.75
        focus_x, focus_y = 0, 0
    else:
        largest = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = largest
        pad = 0.2
        x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
        x2, y2 = min(w, x2 + pad * w), min(h, y2 + pad * h)
        # Clamp to avoid invalid crops
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = w * 0.25, h * 0.25, w * 0.75, h * 0.75
        # Calculate focus offsets for pan
        focus_x = ((x1 + x2) / 2 - w / 2) / w * 100
        focus_y = ((y1 + y2) / 2 - h / 2) / h * 100

    # Start with full panel, then zoom to focus
    full_img = cv2.resize(img, VIDEO_SIZE)  # Full panel first
    focus_img = img[int(y1):int(y2), int(x1):int(x2)]
    focus_img = cv2.resize(focus_img, VIDEO_SIZE)

    # Background blur with cinematic darkening
    bg_img = (fast_blur(img) * 0.7).astype(np.uint8)

    # Create MoviePy clips
    bg_clip = ImageClip(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB), duration=duration)
    full_clip = ImageClip(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB), duration=duration)
    fg_clip = ImageClip(cv2.cvtColor(focus_img, cv2.COLOR_BGR2RGB), duration=duration)

    # Randomized pan/zoom movement with face-targeted pan (bias toward faces)
    zoom_in = True if len(boxes) > 0 else random.choice([True, False])
    start_zoom = 1.0 if zoom_in else random.uniform(1.05, 1.15)
    end_zoom = random.uniform(1.05, 1.15) if zoom_in else 1.0

    start_pos = (random.uniform(-40, 40), random.uniform(-20, 20))
    end_pos = (
        np.clip(-focus_x, -100, 100),
        np.clip(-focus_y, -150, 150)
    )  # pan toward face with safe limits

    # Transition from full panel to zoomed focus with smooth blending
    def zoom_transition(t):
        if t < duration * 0.3:  # First 30%: ease in gently
            ease = (1 - np.cos(np.pi * t / (duration * 0.3))) / 2
            zoom = 1.0 + (start_zoom - 1.0) * ease
            return zoom, (start_pos[0] * ease, start_pos[1] * ease)
        else:  # Rest: zoom to focus
            t_zoom = (t - duration * 0.3)
            return smooth_zoom_pan(t_zoom, duration * 0.7, start_zoom, end_zoom, start_pos, end_pos)

    # Create animated clip from full panel with zoom transition
    animated_clip = full_clip.resized(lambda t: zoom_transition(t)[0])
    animated_clip = animated_clip.with_position(
        lambda t: tuple(sum(x) for x in zip(
            zoom_transition(t)[1],
            handheld_motion(t)
        ))
    )

    # Composite final
    return CompositeVideoClip([bg_clip, animated_clip])

# =========================
# 🎬 Chapter Video Generation
# =========================

def init_yolo():
    global model
    if model is None:
        model = YOLO("yolov8n.pt")

def generate_video_from_folder(folder_path):
    """Creates one full HD manga video from a folder."""
    image_folder = os.path.join(folder_path, "images")

    # Check for audio file (.mp3 or .wav)
    audio_path_mp3 = os.path.join(folder_path, "audio.mp3")
    audio_path_wav = os.path.join(folder_path, "audio.wav")

    if os.path.exists(audio_path_mp3):
        audio_path = audio_path_mp3
    elif os.path.exists(audio_path_wav):
        audio_path = audio_path_wav
    else:
        print(f"⚠️ Skipping {folder_path} — missing images/ or audio.mp3/audio.wav")
        return

    if not os.path.exists(image_folder):
        print(f"⚠️ Skipping {folder_path} — missing images/")
        return

    # Load audio
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration

    # Load and sort images
    images = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if not images:
        print(f"⚠️ No images found in {image_folder}")
        return

    clip_duration = (audio_duration * 0.98) / len(images)  # 2% trim for fade-out safety

    # Build all clips with memory optimization
    clips = []
    for img_path in images:
        clip = create_cinematic_clip(img_path, clip_duration)
        if clip is None:
            continue  # Skip failed images
        clip = clip.with_duration(clip_duration)
        clips.append(clip)
        del clip

    # Dynamic fade length (shorter for short panels)
    fade_time = min(0.4, clip_duration * 0.25)

    # Combine clips with smooth crossfades
    final = concatenate_videoclips(
        clips,
        method="compose",
        padding=-fade_time,
        bg_color=(0, 0, 0)
    )

    final = final.with_audio(audio)

    # Export path
    output_name = os.path.basename(folder_path.rstrip("/")) + ".mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print(f"🎞️ Rendering {output_name} ...")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="h264_videotoolbox",
        bitrate="6000k",
        audio_codec="aac",
        preset="medium",
        threads=4,
        ffmpeg_params=["-movflags", "+faststart"]
    )
    print(f"✅ Exported: {output_path}")

# =========================
# 🚀 Bulk Processing
# =========================

def process_all_chapters(main_dir="manga_project"):
    folders = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    print(f"📁 Found {len(folders)} chapters to render.\n")

    # Use multiple cores for parallel processing
    pool_size = max(1, cpu_count() // 2)  # half cores (safe for MacBook M chips)
    with Pool(pool_size, initializer=init_yolo) as pool:
        list(tqdm(pool.imap(generate_video_from_folder, folders), total=len(folders), desc="Rendering Chapters"))

    # Save YOLO cache once safely after all renders (atomic write)
    tmp_file = CACHE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(yolo_cache, f)
    os.replace(tmp_file, CACHE_FILE)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    process_all_chapters()