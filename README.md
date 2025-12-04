# Mini Video Generator

[![CI](https://github.com/prathmeshmagar447/Mini-Video-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/prathmeshmagar447/Mini-Video-Generator/actions/workflows/ci.yml)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/prathmeshmagar447/Mini-Video-Generator)

A high-performance, cinematic video generator for manga chapters and anime clips optimized for macOS (Apple Silicon). Creates Portrait HD (1080×1920) videos with smooth zoom/pan effects, blurred backgrounds, and dynamic crossfade transitions using AI-powered subject tracking.

**🎯 AI-Powered Focus Detection** - Automatically detects and tracks subjects using YOLOv8 for cinematic camera movements.

## Features

- 📱 **Portrait HD 1080×1920** output (optimized for mobile viewing)
- 🧠 **YOLOv8 AI focus detection** for automatic subject tracking
- 🌫️ **Intelligent background blur** with fast processing
- 🎞️ **Smooth easing zoom/pan** with natural motion and handheld camera effects
- 🔁 **Dynamic crossfade transitions** (adaptive fade timing)
- ⚡ **Multi-core parallel processing** for fast rendering
- 🛡️ **Robust error handling** for corrupted images and invalid detections
- 📁 **Per-image caching** to avoid race conditions in parallel processing
- 🎯 **macOS optimized** for Apple Silicon (M1/M2/M3/M4) with MPS acceleration

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

```
├── manga_project/           # Input directory for manga chapters
│   └── chapter1/           # Chapter folder (can have multiple chapters)
│       ├── images/         # Manga panel images (JPG/PNG)
│       └── audio.wav       # Chapter audio narration (WAV/MP3)
├── output_videos/          # Generated video output directory
├── yolov8n.pt             # YOLOv8 model (auto-downloaded)
├── yolo_cache.json        # Detection cache for performance
├── manga_video_generator.py
├── requirements.txt
└── README.md
```

## Quick Start

1. **Create your manga chapter structure:**
   ```bash
   mkdir -p manga_project/chapter1/images
   # Add your manga images to manga_project/chapter1/images/
   # Add your audio file as manga_project/chapter1/audio.wav or audio.mp3
   ```

2. **Run the generator:**
   ```bash
   python manga_video_generator.py
   ```

3. **Find your video** in the `output_videos/` folder

## Usage

1. **Prepare your manga chapters** in the folder structure above
2. **Run the generator:**
   ```bash
   python manga_video_generator.py
   ```

3. **Find your videos** in the `output_videos/` folder

## Configuration

Edit the settings at the top of `manga_video_generator.py`:

- `VIDEO_SIZE`: Output resolution as (width, height) tuple (default: (1080, 1920) for portrait HD)
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
- Portrait HD 1080×1920 resolution
- H.264 video codec with hardware acceleration
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for AI-powered object detection
- [MoviePy](https://github.com/Zulko/moviepy) for video processing
- [OpenCV](https://opencv.org/) for computer vision operations
