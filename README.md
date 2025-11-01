# Mini Video Generator

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/prathmeshmagar447/Mini-Video-Generator)

A high-performance, cinematic video generator for manga chapters optimized for macOS (Apple Silicon). Creates Full HD (1920×1080) videos with smooth zoom/pan effects, blurred backgrounds, and dynamic crossfade transitions.

**🚀 Sample data included** - Ready to run with pre-processed manga chapter and YOLO models!

## Features

- 🎥 **Full HD 1920×1080** output
- 🧠 **YOLOv8 AI focus detection** for automatic subject tracking
- 🌫️ **Intelligent background blur** with fast processing
- 🎞️ **Smooth easing zoom/pan** with natural motion
- 🔁 **Dynamic crossfade transitions** (adaptive fade timing)
- ⚡ **Multi-core parallel processing** for fast rendering
- 🛡️ **Robust error handling** for corrupted images and invalid detections
- 📁 **Per-image caching** to avoid race conditions in parallel processing
- 🎯 **macOS optimized** for Apple Silicon (M1/M2/M3/M4)

## Requirements

- Python 3.10+ (Apple Silicon optimized)
- FFmpeg (for video encoding)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg (if not already installed):**
   ```bash
   brew install ffmpeg
   ```

3. **Optional: GPU acceleration for YOLOv8**
   ```bash
   pip install torch torchvision torchaudio
   ```

## Project Structure

This repository includes sample data for immediate testing:

```
├── manga_project/
│   └── chapter1/
│       ├── images/
│       │   ├── 0001-023_0..jpg
│       │   ├── 0001-023_1..jpg
│       │   ├── 0001-023_2..jpg
│       │   ├── 0001-023_3..jpg
│       │   ├── 0001-023_4..jpg
│       │   └── 0001-023_5..jpg
│       │   └── *.yolo.json (YOLO detection data)
│       └── audio.wav
├── output_videos/
│   └── chapter1.mp4 (sample generated video)
├── yolov8n.pt (YOLOv8 nano model)
├── yolo_cache.json (detection cache)
├── manga_video_generator.py
├── requirements.txt
└── README.md
```

## Quick Start

**Try it now with sample data:**
```bash
python manga_video_generator.py
```

The script will process the included `chapter1` data and generate a video in `output_videos/`.

## Usage

1. **Prepare your manga chapters** in the folder structure above
2. **Run the generator:**
   ```bash
   python manga_video_generator.py
   ```

3. **Find your videos** in the `output_videos/` folder

## Configuration

Edit the settings at the top of `manga_video_generator.py`:

- `VIDEO_SIZE`: Output resolution (default: 1920×1080)
- `FPS`: Frame rate (default: 24 for cinematic feel)
- `OUTPUT_DIR`: Where videos are saved (default: "output_videos")

## YOLO Model

The script uses YOLOv8n by default. For better manga/anime detection, you can:

1. Download a specialized model (e.g., anime face detector)
2. Replace `"yolov8n.pt"` in the script with your model path

## Performance Tips

- **Multi-core processing**: Automatically uses half your CPU cores
- **Fast blur**: Optimized downscale/upscale blur algorithm
- **Parallel rendering**: Processes multiple chapters simultaneously
- **Memory efficient**: Processes one chapter at a time per core
- **Per-image caching**: Avoids redundant YOLO inference for repeated runs
- **Atomic cache writes**: Prevents corruption of cache files during parallel processing

## Output

Each chapter generates a separate MP4 file with:
- Full HD resolution
- H.264 video codec
- AAC audio codec
- Cinematic 24 FPS
- Smooth transitions and motion

## Troubleshooting

**FFmpeg not found:**
```bash
brew install ffmpeg
```

**Permission issues:**
Make sure you have write permissions in the project directory.

**YOLO model download:**
The script will automatically download yolov8n.pt on first run.

**Memory issues:**
Reduce `VIDEO_SIZE` or process fewer chapters simultaneously.
