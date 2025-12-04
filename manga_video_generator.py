#!/usr/bin/env python3
"""
FFmpeg-backed Motion Manga Engine (improved)

Improvements vs original:
- Vectorized vignette (no per-pixel loops)
- More robust FFmpeg concat (re-encode to libx264 to avoid 'concat: stream parameters mismatch')
- Reproducible random seeding option
- Safer temporary directory (unique per run)
- Fixed transition alpha logic, more consistent durations
- Slightly adjusted handheld/zoom math to avoid extreme crops
- Better multiprocessing progress reporting
- Basic error handling for ffmpeg/ffprobe calls
"""

import os
import cv2
import numpy as np
import math
import subprocess
import shutil
import tempfile
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser
from tqdm import tqdm
from functools import partial
import uuid
import sys

# -----------------------------
# Config (tweak as needed)
# -----------------------------
VIDEO_W, VIDEO_H = 1920, 1080
FPS = 24
BITRATE = "6000k"
SFX_DIR = "sfx"   # Put whoosh.wav, slide.wav, impact.wav here
RANDOM_SEED = None  # set to int for reproducible choices

# -----------------------------
# Utilities
# -----------------------------
def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2

def handheld_motion(t, amp_x=2.5, amp_y=1.5):
    """small handheld jitter over normalized t in [0,1]"""
    return (
        amp_x * math.sin(2 * math.pi * t * 0.5),
        amp_y * math.cos(2 * math.pi * t * 0.6)
    )

def random_pan_vector(rng):
    choices = [
        (50, 0), (-50, 0),
        (0, 50), (0, -50),
        (40, 40), (-40, -40),
        (40, -40), (-40, 40)
    ]
    return choices[rng.integers(0, len(choices))]

def safe_call(cmd):
    """Run subprocess and raise with helpful message if it fails"""
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nExit: {e.returncode}")

# -----------------------------
# Vectorized vignette generator
# -----------------------------
def make_vignette_mask(w=VIDEO_W, h=VIDEO_H, exponent=1.4):
    # center-based radial falloff, returns float32 (h,w,1)
    cx = w / 2.0
    cy = h / 2.0
    xs = (np.arange(w) - cx).astype(np.float32)
    ys = (np.arange(h) - cy).astype(np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)
    max_dist = np.sqrt(cx*cx + cy*cy)
    mask = 1.0 - (dist / max_dist) ** exponent
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask[:, :, np.newaxis]  # shape (h,w,1)

# Precompute vignette once per run for speed
VIGNETTE_MASK = make_vignette_mask()

# -----------------------------
# STYLE PRESET (optimized)
# -----------------------------
def apply_style_preset(img):
    """
    Returns (foreground_frame, background_frame) at (VIDEO_W, VIDEO_H)
    - Foreground: 9:16 crop -> resized, vignette applied, slight bloom
    - Background: stretched to full res, desaturated and teal-graded
    """
    h, w = img.shape[:2]

    # Use FULL IMAGE as foreground
    fg = cv2.resize(img, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)

    # Background: stretch to full frame and desaturate
    bg = cv2.resize(img, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
    # Convert to grayscale then back to BGR (fast)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)

    # Dim & color grade (blue/teal push)
    bg = (bg.astype(np.float32) * 0.65).clip(0, 255).astype(np.uint8)
    # small teal push: increase green slightly, reduce red
    bg = bg.astype(np.float32)
    bg[:, :, 2] *= 0.9  # red channel
    bg[:, :, 1] *= 1.08  # green channel (teal)
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # Apply vignette to foreground using precomputed mask
    fg_float = fg.astype(np.float32) / 255.0
    vignette = VIGNETTE_MASK  # shape (h,w,1)
    # Broadcast vignette to 3 channels
    vignette3 = np.repeat(vignette, 3, axis=2)
    fg_float *= vignette3
    fg_v = (fg_float * 255).astype(np.uint8)

    # Soft bloom: small gaussian blur blended in
    blur = cv2.GaussianBlur(fg_v, (0, 0), 12)
    bloom = cv2.addWeighted(fg_v, 0.8, blur, 0.2, 0)

    return bloom, bg

# -----------------------------
# Per-image Segment Renderer
# -----------------------------
def render_segment(task, fps=FPS):
    """
    task: (img_path, duration, out_path, seed)
    Returns out_path on success, None on failure.
    """
    img_path, duration, out_path, seed = task
    rng = np.random.default_rng(seed)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[render_segment] Failed to load {img_path}", file=sys.stderr)
            return None

        fg, bg = apply_style_preset(img)

        dx, dy = random_pan_vector(rng)
        start_zoom = 1.0
        end_zoom = float(rng.uniform(1.03, 1.15))  # a bit less extreme for safety

        frames = max(1, int(round(duration * fps)))
        # ensure frames >=1 and consistent timing; use FPS from caller
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (VIDEO_W, VIDEO_H))

        for i in range(frames):
            frame_fg = fg.copy()
            writer.write(frame_fg)

        writer.release()
        return out_path

    except Exception as e:
        print(f"[render_segment] Error {img_path}: {e}", file=sys.stderr)
        return None

# -----------------------------
# TRANSITION EFFECTS (fixed)
# -----------------------------
def generate_white_flash(path, frames=3, fps=FPS):
    white = np.full((VIDEO_H, VIDEO_W, 3), 255, dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (VIDEO_W, VIDEO_H))
    for _ in range(frames):
        out.write(white)
    out.release()

def generate_swipe_transition(path, frames=6, fps=FPS):
    # swipe: blurred teal-ish frames with progressive blur
    base = np.full((VIDEO_H, VIDEO_W, 3), (200, 220, 255), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (VIDEO_W, VIDEO_H))
    for i in range(frames):
        blur_amt = 1 + int((i / max(1, frames - 1)) * 35)
        swipe = cv2.GaussianBlur(base, (0, 0), blur_amt)
        out.write(swipe)
    out.release()

def generate_speedline_transition(path, frames=5, fps=FPS):
    # speedlines: draw semi-random slanted lines, increase opacity over frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (VIDEO_W, VIDEO_H))
    for i in range(frames):
        img = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        for _ in range(40):
            x = np.random.randint(-200, VIDEO_W + 200)
            x2 = x + np.random.randint(120, 260)
            cv2.line(img, (x, 0), (x2, VIDEO_H), (200, 220, 255), 2)
        alpha = (i + 1) / frames
        img = (img.astype(np.float32) * alpha).astype(np.uint8)
        out.write(img)
    out.release()

def create_transition_segment(i, temp_dir):
    path = os.path.join(temp_dir, f"transition_{i}_{uuid.uuid4().hex[:8]}.mp4")
    effect = np.random.choice(["white", "swipe", "speed"])
    if effect == "white":
        generate_white_flash(path)
    elif effect == "swipe":
        generate_swipe_transition(path)
    else:
        generate_speedline_transition(path)
    return path

# -----------------------------
# FFmpeg Concat with SFX (robust)
# -----------------------------
def pick_sfx():
    if not os.path.isdir(SFX_DIR):
        return None
    files = [f for f in os.listdir(SFX_DIR) if f.lower().endswith(".wav")]
    if not files:
        return None
    return os.path.join(SFX_DIR, np.random.choice(files))

def ffmpeg_concat(segment_paths, audio_path, output_path, temp_dir):
    """
    Create a concat list, re-encode merged result with libx264 (to avoid mismatch)
    Then add original audio and optional sfx mix. This ensures concat reliability.
    """
    final_list = []
    for i, seg in enumerate(segment_paths):
        final_list.append(seg)
        if i < len(segment_paths) - 1:
            final_list.append(create_transition_segment(i, temp_dir))

    listfile = os.path.join(temp_dir, "segments.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in final_list:
            f.write(f"file '{os.path.abspath(p)}'\n")

    merged = os.path.join(temp_dir, "merged.mp4")
    # Re-encode to a consistent codec/params — libx264, yuv420p
    cmd_merge = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", merged
    ]
    safe_call(cmd_merge)

    # Add main audio (audio_path may already be mp3/wav)
    final_audio = os.path.join(temp_dir, "with_audio.mp4")
    cmd_audio = [
        "ffmpeg", "-y", "-i", merged, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", final_audio
    ]
    safe_call(cmd_audio)

    sfx = pick_sfx()
    if sfx:
        # Mix original audio and sfx
        cmd_mix = [
            "ffmpeg", "-y", "-i", final_audio, "-i", sfx,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:weights=1 1:duration=shortest[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", output_path
        ]
        safe_call(cmd_mix)
    else:
        shutil.copy(final_audio, output_path)

# -----------------------------
# High-level Pipeline
# -----------------------------
def process_folder(folder_path, out_dir, fps=FPS, rng=None, anime_mode=False):
    temp_dir = os.path.join(tempfile.gettempdir(), f"_manga_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        audio_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.mp3', '.wav'))
        ]
        audio_path = os.path.join(folder_path, audio_files[0]) if audio_files else None

        if anime_mode:
            media = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ])
        else:
            media = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov", ".mkv"))
            ])
        if not media:
            print(f"No media found in {folder_path}")
            return

        if audio_path is None:
            print(f"Skipping {folder_path}: no audio file.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Probe audio duration
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ])
            audio_duration = float(probe.decode().strip())
        except Exception as e:
            print(f"ffprobe failed for {audio_path}: {e}", file=sys.stderr)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        clip_duration = (audio_duration * 0.98) / len(media) if not anime_mode else None

        # Helper to get duration from video file
        def get_video_duration(video_path):
            try:
                probe = subprocess.check_output([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path
                ])
                return float(probe.decode().strip())
            except:
                return 0.0

        segment_paths = []
        tasks = []

        for i, media_path in enumerate(media):
            seg_path = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
            if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                # it's an image
                seed = rng.integers(0, 2**31 - 1) if rng is not None else np.random.randint(0, 2**31 - 1)
                tasks.append((media_path, clip_duration, seg_path, seed))
                segment_paths.append(seg_path)
            else:
                # it's a video, process directly using ffmpeg to resize and clip (if not anime_mode)
                # First, check if video is readable
                try:
                    probe = subprocess.check_output([
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        media_path
                    ])
                    vid_duration = float(probe.decode().strip())
                    if vid_duration <= 0:
                        raise ValueError("Invalid duration")
                except Exception as e:
                    print(f"Skipping corrupted or invalid video: {media_path} ({e})")
                    continue

                cmd = [
                    "ffmpeg", "-y", "-i", media_path
                ]
                if clip_duration is not None:
                    cmd += ["-t", str(clip_duration)]
                cmd += [
                    "-vf", f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2",
                    "-r", str(fps),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    seg_path
                ]
                try:
                    safe_call(cmd)
                    segment_paths.append(seg_path)
                except RuntimeError as e:
                    print(f"Failed to process video: {media_path} ({e})")
                    continue

        pool_size = max(1, int(cpu_count() / 2))
        print(f"Rendering {len(tasks)} panels using {pool_size} workers...")
        # Use partial to ensure fps is used
        with Pool(pool_size) as pool:
            for _ in tqdm(pool.imap_unordered(partial(render_segment, fps=fps), tasks), total=len(tasks)):
                pass

        # Calculate total video duration and repeat clips if shorter than audio
        total_video_duration = sum(get_video_duration(seg) for seg in segment_paths if seg in segment_paths)
        if total_video_duration < audio_duration:
            print(f"Video duration ({total_video_duration:.2f}s) < audio duration ({audio_duration:.2f}s), repeating clips...")
            original_segments = segment_paths[:]
            additional_segments = []
            while total_video_duration < audio_duration:
                next_seg = original_segments[len(additional_segments) % len(original_segments)]
                additional_segments.append(next_seg)
                total_video_duration += get_video_duration(next_seg)
            segment_paths.extend(additional_segments)
            print(f"Added {len(additional_segments)} repeated clips, new duration: {total_video_duration:.2f}s")

        out_name = os.path.basename(folder_path.rstrip("/\\"))
        out_name = out_name + ".mp4"
        out_path = os.path.join(out_dir, out_name)

        print("Concatenating with transitions + audio...")
        ffmpeg_concat(segment_paths, audio_path, out_path, temp_dir)

        print(f"✔ Exported: {out_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--project", default="manga_project")
    parser.add_argument("--out", default="output_videos")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--anime", action="store_true", help="Anime mode: process video clips without duration clipping")
    args = parser.parse_args()

    if args.seed is not None:
        global RANDOM_SEED
        RANDOM_SEED = int(args.seed)

    rng = np.random.default_rng(RANDOM_SEED)

    os.makedirs(args.out, exist_ok=True)

    chapters = []
    for root, dirs, files in os.walk(args.project):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov', '.mkv')) for f in files):
            chapters.append(root)

    print(f"Found {len(chapters)} chapters.")
    for chapter in chapters:
        process_folder(chapter, args.out, fps=args.fps, rng=rng, anime_mode=args.anime)

if __name__ == "__main__":
    main()
