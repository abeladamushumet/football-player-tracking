# ⚽ Football Player Tracking System

A comprehensive computer vision system for detecting, tracking, and analyzing football players in match videos using YOLOv8 and advanced tracking algorithms.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **🔍 Player Detection**: YOLOv8-based real-time player detection with CPU optimization
- **🎯 Multi-Object Tracking**: ByteTrack integration for robust player tracking across frames
- **�️ dField Mapping**: Homography transformation to map player positions to real-world field coordinates (FIFA standard 105m × 68m)
- **📊 Advanced Visualizations**:
  - Player movement heatmaps with Gaussian filtering
  - Trajectory plots with distance calculations
  - Distance ranking analysis for performance metrics
- **⚡ CPU-Friendly**: Optimized for CPU processing with frame skipping and dynamic resizing
- **📈 Performance Analytics**: Distance calculation and ranking visualization

## 🎬 Demo

```
Input: Football match video (MP4)
   ↓
Detection: YOLOv8 identifies players
   ↓
Tracking: ByteTrack assigns persistent IDs
   ↓
Mapping: Homography transforms to field coordinates
   ↓
Output: Annotated video + Heatmaps + Trajectories + Analytics
```

---

## 📁 Project Structure

```
football-player-tracking/
├── 📂 data/
│   ├── raw/                          # Raw video files
│   │   └── 1.mp4                     # Input match video (place your video here)
│   └── processed/
│       └── frames/                   # Extracted frames (2020+ frames @ 5-frame intervals)
│           ├── frame_00005.jpg
│           ├── frame_00010.jpg
│           └── ... (frame_10100.jpg)
│
├── 📂 models/
│   └── detection/
│       └── yolov8/                   # YOLOv8 model weights
│           ├── yolov8n.pt           # Nano model (6MB, fastest)
│           └── yolov8m.pt           # Medium model (52MB, more accurate)
│
├── 📂 src/
│   ├── detection/
│   │   └── detector.py              # 🔍 YOLOv8 player detection with CPU optimization
│   ├── tracking/
│   │   └── tracker.py               # 🎯 Multi-object tracking with ByteTrack + homography
│   ├── homography/
│   │   ├── field_mapping.py         # 🗺️ Field coordinate transformation
│   │   └── transform_utils.py       # Homography matrix computation utilities
│   ├── preprocessing/
│   │   ├── video_loader.py          # 📹 Video loading with metadata extraction
│   │   └── frame_extractor.py       # 🎞️ Frame extraction from video
│   └── visualization/
│       ├── heatmap.py               # 🔥 Player heatmap generation (Gaussian filtered)
│       ├── trajectory_plot.py       # 📈 Trajectory visualization with smoothing
│       └── distance_ranking.py      # 📊 Distance covered analysis & ranking
│
├── 📂 outputs/
│   ├── videos/                      # Processed videos with annotations
│   │   ├── output.mp4              # Detection output
│   │   └── tracking_output.avi     # Tracking output with IDs
│   ├── heatmaps/                    # Generated heatmap images
│   │   ├── heatmap_ID_*.png        # Individual player heatmaps
│   ├── tracking_output.json         # Tracking data (pixel coordinates)
│   ├── tracking_field_coords.json   # Field-mapped coordinates (meters)
│   ├── trajectory_plot.png          # Trajectory visualization
│   └── distance_ranking.png         # Distance ranking chart
│
├── 📂 tests/
│   └── test_video_loader.py        # Unit tests for video loading
│
├── 📄 requirements.txt              # Python dependencies (5 packages)
├── 📄 .gitignore                    # Git ignore rules
├── � LICENSE                     # Apache License
└── 📄 README.md                     # This file
```

### Key Files Description

| File | Purpose | Key Features |
|------|---------|--------------|
| `detector.py` | Player detection | CPU-optimized, frame skipping, confidence filtering |
| `tracker.py` | Player tracking | ByteTrack, persistent IDs, homography mapping |
| `field_mapping.py` | Coordinate transformation | RANSAC-based homography, real-world mapping |
| `transform_utils.py` | Homography utilities | Matrix computation, point transformation |
| `video_loader.py` | Video loading | Metadata extraction, validation |
| `frame_extractor.py` | Frame extraction | Configurable skip rate, batch processing |
| `heatmap.py` | Heatmap generation | Gaussian filtering, top-N players, FIFA field |
| `trajectory_plot.py` | Trajectory visualization | Smoothing, distance calculation, color-coded |
| `distance_ranking.py` | Performance analytics | Distance ranking, bar chart visualization |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone the Repository

```bash
https://github.com/abeladamushumet/football-player-tracking.git
cd football-player-tracking
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `numpy` - Numerical computing
- `scipy` - Scientific computing (Gaussian filtering)
- `opencv-python` - Computer vision operations
- `ultralytics` - YOLOv8 implementation
- `matplotlib` - Visualization and plotting

### Step 4: Download YOLOv8 Models

The YOLOv8 models will be automatically downloaded on first run, or you can manually download:

```bash
# Create models directory
mkdir -p models/detection/yolov8

# Download YOLOv8 nano (fastest, ~6MB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/detection/yolov8/

# Download YOLOv8 medium (more accurate, ~52MB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P models/detection/yolov8/
```

**Windows users (without wget):**
```bash
# Using PowerShell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "models/detection/yolov8/yolov8n.pt"
```

### Step 5: Prepare Your Data

Place your football match video in the `data/raw/` directory:

```bash
# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path data\raw, data\processed\frames, outputs\videos, outputs\heatmaps

# Copy your video file (rename to 1.mp4 or update paths in scripts)
# Windows:
copy "C:\path\to\your\video.mp4" data\raw\1.mp4

```

**Supported Video Formats:**
- MP4 (recommended)
- AVI
- MOV
- MKV

**Video Requirements:**
- Clear view of football field
- Visible field lines for homography calibration
- Resolution: 720p or higher recommended
- Duration: Any (processing time scales linearly)

---

## 📖 Usage Guide

### 🚀 Quick Start (Complete Pipeline)

Run the complete tracking pipeline in order:

```bash
# Navigate to project root
cd football-player-tracking

# 1. Extract frames 
python src/preprocessing/frame_extractor.py

# 2. Run detection 
python src/detection/detector.py

# 3. Run tracking (MAIN PIPELINE - includes detection + tracking + homography)
python src/tracking/tracker.py

# 4. Generate visualizations (run after tracking completes)
python src/visualization/heatmap.py
python src/visualization/trajectory_plot.py
python src/visualization/distance_ranking.py
```

**⏱️ Estimated Processing Time:**
- Frame extraction: ~2-5 minutes (for 10-minute video)
- Detection: ~10-15 minutes (skip_frames=5, CPU)
- Tracking: ~15-20 minutes (includes detection + tracking)
- Visualizations: ~1-2 minutes each

**💡 Pro Tip:** Start with tracking (step 3) as it includes detection. Use detection separately only for testing different parameters.

---

### 1. Video Loading and Frame Extraction

Extract frames from your match video for analysis or debugging:

```bash
python src/preprocessing/frame_extractor.py
```

**Configuration in `frame_extractor.py` (lines 40-42):**
```python
video_path = "data/raw/1.mp4"        # Input video path
output_dir = "data/processed/frames"  # Output directory
skip_frames = 5                       # Extract every 5th frame
```

**Expected Output:**
- Extracted frames saved to `data/processed/frames/`
- Frame naming: `frame_00005.jpg`, `frame_00010.jpg`, ..., `frame_10100.jpg`
- Console output: `✅ Finished extracting X frames to data/processed/frames`

**Use Cases:**
- Visual inspection of video quality
- Manual annotation for ground truth
- Debugging detection/tracking issues
- Creating training datasets

**Storage Note:** ~2000 frames = ~500MB disk space

---

### 2. Player Detection (Standalone)

Run YOLOv8 detection on video (standalone mode for testing):

```bash
python src/detection/detector.py
```

**Configuration in `detector.py` (lines 95-103):**
```python
detector = PlayerDetectorCPU(
    model_path="models/detection/yolov8/yolov8n.pt",  # Model selection
    output_dir="outputs/videos",                       # Output directory
    conf_thresh=0.4,                                   # Confidence threshold
    skip_frames=5,                                     # Process every 5th frame
    resize_width=640                                   # Resize for CPU optimization
)
```

**Features:**
- ✅ CPU-optimized processing with frame skipping
- ✅ Configurable confidence threshold (default: 0.4)
- ✅ Automatic frame resizing for faster processing
- ✅ Real-time progress updates every 100 frames
- ✅ JSON output with detection metadata

**Output Files:**
- `outputs/videos/output.mp4` - Video with bounding boxes (green rectangles)
- `outputs/detections.json` - Detection data per frame (bbox, confidence, class)

**Performance Tuning:**

| Parameter | Fast | Balanced | Accurate |
|-----------|------|----------|----------|
| `model_path` | yolov8n.pt | yolov8n.pt | yolov8m.pt |
| `skip_frames` | 10 | 5 | 2 |
| `resize_width` | 480 | 640 | 1280 |
| `conf_thresh` | 0.5 | 0.4 | 0.3 |
| **Speed** | ~20 FPS | ~15 FPS | ~5 FPS |

**When to Use:**
- Testing different detection parameters
- Evaluating model performance
- Creating detection-only datasets
- Debugging detection issues

**Note:** For full pipeline, use `tracker.py` instead (includes detection + tracking).

---

### 3. Player Tracking (Main Pipeline) ⭐

Track players across frames with unique IDs and field mapping:

```bash
python src/tracking/tracker.py
```

**This is the MAIN pipeline that includes:**
1. ✅ YOLOv8 detection (person class only)
2. ✅ ByteTrack multi-object tracking
3. ✅ Persistent ID assignment
4. ✅ Homography transformation to field coordinates
5. ✅ JSON output for downstream analysis

**Configuration in `tracker.py`:**

```python
# Line 10-13: Output paths
self.video_path = "data/raw/1.mp4"
self.output_video_path = "outputs/videos/tracking_output.avi"
self.output_json_path = "outputs/tracking_output.json"
self.output_field_json = "outputs/tracking_field_coords.json"

# Line 18: Model selection
self.model = YOLO("models/detection/yolov8/yolov8m.pt")  # Medium model

# Line 24-35: Homography calibration points (MUST UPDATE!)
image_points = [
    (100, 200),     # top-left corner of field
    (1800, 220),    # top-right corner of field
    (150, 900),     # bottom-left corner of field
    (1750, 880)     # bottom-right corner of field
]

# Line 37-44: Real-world field coordinates (FIFA standard)
field_points = [
    (0, 0),         # top-left (meters)
    (105, 0),       # top-right
    (0, 68),        # bottom-left
    (105, 68)       # bottom-right
]
```

**Output Files:**
- `outputs/videos/tracking_output.avi` - Video with player IDs (green boxes + ID labels)
- `outputs/tracking_output.json` - Tracking data in pixel coordinates
- `outputs/tracking_field_coords.json` - Field coordinates in meters (for visualizations)


### 4. Visualization

#### Generate Player Heatmaps

```bash
cd src/visualization
python heatmap.py
```

Creates individual heatmaps for top 5 most active players showing their movement patterns on the field.

**Output:** `outputs/heatmaps/heatmap_ID_*.png`

**Customization:**
- Edit `MIN_FRAMES` to filter players with fewer appearances
- Edit `TOP_N` to change number of heatmaps generated
- Heatmaps use Gaussian filtering for smooth visualization

---

#### Generate Trajectory Plot

```bash
python trajectory_plot.py
```

Visualizes movement trajectories of top 10 players with distance calculations.

**Output:** `outputs/trajectory_plot.png`

**Features:**
- Color-coded trajectories for each player
- Distance covered displayed in legend
- Smoothed trajectories for cleaner visualization
- FIFA-standard field dimensions (105m × 68m)

---

#### Generate Distance Ranking

```bash
python distance_ranking.py
```

Creates a bar chart ranking players by total distance covered.

**Output:** `outputs/distance_ranking.png`

**Features:**
- Top 10 players by distance
- Filters players with < 300 frames (noise reduction)
- Distance calculated in meters

---

### 5. Testing

Run unit tests:

```bash
cd tests
python test_video_loader.py
```

**Expected Output:**
```
✅ Video loaded successfully.
📹 Video Information:
   Resolution: 1920 x 1080
   FPS: 25.0
   Total Frames: 15000
```

---

## 🔧 Configuration

### Detection Parameters

Edit `src/detection/detector.py`:

```python
detector = PlayerDetectorCPU(
    model_path="models/detection/yolov8/yolov8n.pt",
    output_dir="outputs/videos",
    conf_thresh=0.4,      # Detection confidence threshold
    skip_frames=5,        # Process every 5th frame
    resize_width=640      # Resize width for faster processing
)
```

### Tracking Parameters

Edit `src/tracking/tracker.py`:

```python
# Tracker configuration
self.model = YOLO("models/detection/yolov8/yolov8m.pt")

# ByteTrack settings
results = self.model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0]  # 0 = person class
)
```

### Visualization Parameters

Edit constants in visualization scripts:

**`heatmap.py`:**
```python
FIELD_LENGTH = 105      # Field length in meters
FIELD_WIDTH = 68        # Field width in meters
MIN_FRAMES = 200        # Minimum frames to include player
TOP_N = 5               # Number of players to visualize
```

**`trajectory_plot.py`:**
```python
MIN_FRAMES = 120        # Minimum frames for trajectory
TOP_N_TRACKS = 10       # Number of trajectories to plot
SMOOTHING_WINDOW = 5    # Trajectory smoothing window
```

---

## 📊 Output Examples

### Detection Output (JSON)
```json
[
    {
        "frame": 5,
        "class_id": 0,
        "confidence": 0.89,
        "bbox": [245, 180, 298, 356]
    }
]
```

### Tracking Output (JSON)
```json
[
    {
        "frame_id": 0,
        "tracks": [
            {
                "track_id": 1,
                "bbox": [245, 180, 298, 356]
            }
        ]
    }
]
```

### Field Coordinates (JSON)
```json
[
    {
        "frame_id": 0,
        "field_tracks": [
            {
                "track_id": 1,
                "field_pos": [52.3, 34.1]
            }
        ]
    }
]
```

---

## 🎓 How It Works

### 1. Detection Pipeline

```
Video Input → Frame Extraction → YOLOv8 Detection → Bounding Boxes
```

- YOLOv8 detects persons (class 0) in each frame
- Confidence filtering removes low-quality detections
- Bounding boxes saved with frame numbers

### 2. Tracking Pipeline

```
Detections → ByteTrack → Persistent IDs → Homography → Field Coordinates
```

- ByteTrack assigns consistent IDs across frames
- Handles occlusions and re-identifications
- Homography maps pixel coordinates to real-world meters

### 3. Homography Transformation

```
Image Points (pixels) → Homography Matrix → Field Points (meters)
```

- Uses 4+ point correspondences between image and field
- RANSAC algorithm for robust estimation
- Bottom-center of bbox represents player position (feet)

### 4. Visualization Pipeline

```
Field Coordinates → Filtering → Smoothing → Plotting
```

- Filters short/noisy tracks (< MIN_FRAMES)
- Gaussian smoothing for heatmaps
- Moving average for trajectory smoothing

---
