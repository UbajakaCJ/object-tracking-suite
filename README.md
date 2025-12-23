# 🎯 Object Tracking Suite
**Real-time Object Detection and Tracking with YOLOv11**

A comprehensive Python toolkit for real-time object detection and tracking using YOLOv11, featuring single-stream tracking, multi-threaded processing, and trajectory visualization capabilities.

## 🎯 Features

### Core Functionality
- **Single Stream Tracking**: Real-time object detection and tracking on video files or webcam feeds
- **Multi-threaded Processing**: Concurrent tracking across multiple video streams with different YOLO models
- **Trajectory Visualization**: Track and visualize object movement paths over time with trail rendering
- **Model Flexibility**: Support for detection (`yolo11n.pt`) and segmentation (`yolo11n-seg.pt`) models
- **Live Display**: Real-time visualization of tracking results with OpenCV

### Technical Highlights
- **Framework**: Ultralytics YOLOv11 with OpenCV integration
- **Performance**: GPU-accelerated inference with CUDA support
- **Tracking**: Persistent object ID assignment across frames
- **Visualization**: Annotated bounding boxes, segmentation masks, and movement trails
- **Threading**: Parallel processing for handling multiple video sources simultaneously

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for real-time performance)
- Webcam (optional, for live feed tracking)

### Installation

1. **Clone or download the project files**
```bash
cd object-tracking-suite
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

The requirements include:
- `ultralytics` - YOLOv11 framework
- `lap` - Linear Assignment Problem solver for tracking

3. **Prepare video resources**
```bash
mkdir -p Resources/Videos
# Place your video files in Resources/Videos/
```

### Running the Applications

**Option 1: Basic Object Tracking (Single Stream)**

Track objects in a video file or webcam feed:
```bash
python scripts/objectTracking.py
```
- Uses `yolo11n.pt` model
- Source: `Resources/Videos/video5.mp4` (modify in code for webcam: `cap = cv2.VideoCapture(0)`)
- Press `q` to quit

**Option 2: Tracking with Trail Visualization**

Track objects and visualize their movement paths:
```bash
python scripts/objecttracking_trails.py
```
- Displays 30-frame movement trails for each tracked object
- Trail color: Blue (RGB: 230, 0, 0)
- Source: `Resources/Videos/video7.mp4`
- Press `w` to quit

**Option 3: Multi-threaded Tracking**

Process multiple video streams simultaneously:
```bash
python scripts/multithreaded_tracking.py
```
- Concurrent processing of 2 video streams
- Uses different models: `yolo11n.pt` and `yolo11n-seg.pt`
- Sources: `video5.mp4` and `video8.mp4`
- Automatically saves results

**Option 4: Object Counting (Line-crossing)**

Count objects entering/exiting a designated area:
```bash
python scripts/object_counting.py --video Resources/Videos/video7.mp4 --orientation horizontal --pos 0.5 --space-side below
```
- Configurable reference line orientation and position
- Uses persistent track IDs to avoid double-counting
- Saves annotated output video (default: `output_videos/tracked_count.mp4`)

## 📊 Scripts Explained

### objectTracking.py
Basic single-stream object tracking implementation.

**Features:**
- Frame-by-frame processing
- Persistent object ID tracking
- Real-time visualization
- Simple quit mechanism (press 'q')

**Use Cases:**
- Testing YOLO models on video files
- Live webcam monitoring
- Basic object detection tasks

**Configuration:**
```python
model = YOLO("yolo11n.pt")  # Change model here
cap = cv2.VideoCapture("Resources/Videos/video5.mp4")  # Change source
```

### objecttracking_trails.py
Advanced tracking with trajectory visualization.

**Features:**
- Track history storage (up to 30 frames per object)
- Polyline trail rendering
- Center-point based tracking
- Movement pattern visualization

**Use Cases:**
- Analyzing object movement patterns
- Traffic flow analysis
- Crowd movement studies
- Sports analytics

**Configuration:**
```python
track_history = defaultdict(lambda: [])  # Stores track history
if len(track) > 30:  # Adjust trail length here
    track.pop(0)
cv2.polylines(..., thickness=10)  # Adjust trail thickness
```

### object_counting.py
Object counting via reference line crossing.

**Features:**
- Counts objects entering or exiting a space by crossing a configurable reference line
- Uses persistent tracker IDs to reduce double-counting
- Configurable orientation (`horizontal`/`vertical`), relative position (`--pos`), and `space-side`
- Debounce frames to prevent rapid re-counting around the line
- Saves annotated output video to `output_videos` by default

**Use Cases:**
- People counting at doorways
- Vehicle counting across lanes
- Access control / zone monitoring

**Usage:**
```bash
python scripts/object_counting.py --video Resources/Videos/video7.mp4 --model yolo11n.pt --orientation horizontal --pos 0.5 --space-side below --out output_videos/counts.mp4
```

**Configuration:**
- `--orientation`: `horizontal` or `vertical` (default: `horizontal`)
- `--pos`: Relative line position between 0.0 and 1.0 (default: `0.8`)
- `--space-side`: Which side is considered the monitored space (`above`, `below`, `left`, `right`)
- `--debounce`: Number of frames to debounce a count (default: 5)

### multithreaded_tracking.py
Parallel processing for multiple video streams.

**Features:**
- Thread-based concurrent processing
- Multiple model support
- Daemon threads for clean shutdown
- Automatic result saving

**Use Cases:**
- Multi-camera surveillance systems
- Comparing different YOLO models
- High-throughput video processing
- Distributed monitoring applications

**Configuration:**
```python
MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]  # Add more models
SOURCES = ["Resources/Videos/video5.mp4", "Resources/Videos/video8.mp4"]  # Add more sources
```

## 🎨 Visualization Features

### Bounding Boxes
- Color-coded detection boxes
- Class labels with confidence scores
- Real-time annotation updates

### Segmentation Masks (yolo11n-seg.pt)
- Pixel-level object segmentation
- Transparent mask overlays
- Enhanced object boundary definition

### Movement Trails
- Persistent trajectory lines
- Configurable trail length (default: 30 frames)
- Color-coded paths
- Center-point tracking

## 🔧 Configuration

### Model Selection

Available YOLOv11 models (nano to extra-large):
```python
# Detection models
model = YOLO("yolo11n.pt")   # Nano - fastest
model = YOLO("yolo11s.pt")   # Small
model = YOLO("yolo11m.pt")   # Medium
model = YOLO("yolo11l.pt")   # Large
model = YOLO("yolo11x.pt")   # Extra-large - most accurate

# Segmentation models
model = YOLO("yolo11n-seg.pt")  # Nano segmentation
model = YOLO("yolo11s-seg.pt")  # Small segmentation
# ... and so on
```

### Video Sources

```python
# Video file
cap = cv2.VideoCapture("path/to/video.mp4")

# Webcam (default camera)
cap = cv2.VideoCapture(0)

# External camera
cap = cv2.VideoCapture(1)

# RTSP stream
cap = cv2.VideoCapture("rtsp://camera_ip:port/stream")
```

### Tracking Parameters

```python
# Basic tracking
results = model.track(frame, persist=True)

# Advanced tracking with confidence threshold
results = model.track(
    frame,
    persist=True,
    conf=0.5,      # Confidence threshold
    iou=0.7,       # IOU threshold for NMS
    tracker="bytetrack.yaml"  # Tracker configuration
)
```

## 📁 Project Structure


```
object-tracking-suite/
├── model/
    ├── yolo11n-seg.pt                 # YOLOv11 Nano segmentation model
    └── yolo11n.pt                     # YOLOv11 Nano detection model
├── Resources/
    └── Videos/
        ├── video5.mp4                 # Sample video 1
        ├── video7.mp4                 # Sample video 2
        └── video8.mp4                 # Sample video 3
├── scripts/
    ├── multithreaded_tracking.py      # Multi-stream concurrent tracking
    ├── object_counting.py             # Line-crossing object counting
    ├── objecttracking_trails.py       # Tracking with trail visualization
    └── objectTracking.py              # Basic single-stream tracking
├── README.md                          # This file
└── requirements.txt                   # Python dependencies


```

## 🎬 Workflow Examples

### Campus Monitoring System
```python
# objectTracking.py modification
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True, conf=0.6)
        annotated_frame = results[0].plot()
        cv2.imshow("Campus Monitor", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

### Traffic Analysis with Trails
```python
# objecttracking_trails.py modification
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("traffic_footage.mp4")
# Trail length = 50 frames for longer paths
if len(track) > 50:
    track.pop(0)
```

### Multi-Camera Surveillance
```python
# multithreaded_tracking.py modification
MODEL_NAMES = ["yolo11n.pt", "yolo11n.pt", "yolo11n.pt", "yolo11n.pt"]
SOURCES = [
    "rtsp://camera1/stream",
    "rtsp://camera2/stream", 
    "rtsp://camera3/stream",
    "rtsp://camera4/stream"
]
```

## 🚀 Performance Tips

### GPU Acceleration
Ensure CUDA is properly installed for GPU acceleration:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Optimize for Real-time Processing
```python
# Reduce input resolution for faster processing
results = model.track(frame, imgsz=640)  # Default: 640
results = model.track(frame, imgsz=320)  # Faster but less accurate

# Limit detections per frame
results = model.track(frame, max_det=100)  # Max 100 detections
```

### Memory Management
```python
# Clear GPU cache periodically in long-running applications
import torch
torch.cuda.empty_cache()
```

## 🔮 Future Enhancements

### Phase 2
- [ ] Web-based dashboard for remote monitoring
- [ ] Object counting and statistics
- [ ] Zone-based intrusion detection
- [ ] Export tracking data to CSV/JSON
- [ ] Custom object class filtering

### Phase 3
- [ ] Database integration for historical tracking data
- [ ] REST API for integration with other systems
- [ ] Email/SMS alerts for specific detections
- [ ] Heat map visualization of object density
- [ ] Cloud deployment with Docker/Kubernetes

### Phase 4
- [ ] Re-identification across multiple cameras
- [ ] Behavior analysis and anomaly detection
- [ ] Integration with access control systems
- [ ] Mobile app for remote viewing
- [ ] Advanced analytics dashboard

## 🛠 Known Limitations (Current Version)

1. **Model Download**: First run downloads YOLO models (~6-100MB depending on variant)
2. **GPU Memory**: Larger models require more VRAM (4GB+ recommended)
3. **Real-time Performance**: Depends on hardware; may need frame skipping on CPU-only systems
4. **Track Persistence**: Object IDs may be reassigned if objects leave and re-enter frame
5. **Recording**: Most scripts display results live; multithreaded tracking and `object_counting.py` also save annotated output videos by default, while others can be adapted to save as needed.

## 🤝 Extending the Project

### Adding Custom Models

Train your own YOLOv11 model and use it:
```python
# Train on custom dataset
model = YOLO("yolo11n.pt")
model.train(data="custom_data.yaml", epochs=100)

# Use trained model
model = YOLO("runs/detect/train/weights/best.pt")
```

### Custom Tracking Logic

Implement custom tracking algorithms:
```python
from collections import defaultdict

track_data = defaultdict(dict)

for box, track_id in zip(boxes, track_ids):
    if track_id not in track_data:
        track_data[track_id]['first_seen'] = frame_count
        track_data[track_id]['class'] = class_name
    
    track_data[track_id]['last_seen'] = frame_count
    track_data[track_id]['positions'].append((x, y))
```

### Integration with Other Systems

```python
# Send detection data to external API
import requests

for result in results:
    if result.boxes.id is not None:
        detection_data = {
            'timestamp': time.time(),
            'objects': len(result.boxes),
            'classes': result.boxes.cls.tolist()
        }
        requests.post('http://your-api/detections', json=detection_data)
```

## 📄 License

MIT License - feel free to use and modify for your projects

## 🙋 Support

For issues or questions:
- Check that video files exist in `Resources/Videos/`
- Ensure YOLO models are downloaded (automatic on first run)
- Verify OpenCV is displaying windows correctly
- Check GPU availability for optimal performance

## 🎓 Learning Resources

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [YOLO Object Tracking Guide](https://docs.ultralytics.com/modes/track/)
- [Computer Vision Best Practices](https://github.com/ultralytics/ultralytics)

## 🏆 Use Cases

This toolkit is ideal for:
- **Security & Surveillance**: Monitor restricted areas, detect intrusions
- **Traffic Management**: Vehicle counting, speed estimation, flow analysis
- **Retail Analytics**: Customer tracking, queue management, heat mapping
- **Sports Analysis**: Player tracking, movement pattern analysis
- **Wildlife Monitoring**: Animal detection and behavior study
- **Industrial Automation**: Defect detection, assembly line monitoring


