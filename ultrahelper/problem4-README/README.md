# Parallel Inference Pipeline

## Overview
This repository implements a **parallel inference pipeline** for real‑time pose estimation on video. The pipeline is divided into two components:

1. **Hardware model**: Runs on the GPU and performs the bulk of the convolutional operations (`load_hardware_model()`).
2. **Postprocessor**: Runs on the CPU to decode raw outputs, apply Non‑Maxima Suppression (NMS), draw bounding boxes and keypoints, and compute performance metrics (`load_postprocessor()`).

The pipeline operates in an infinite loop, streaming frames from a video source, processing them in parallel, and displaying:
- **Frame rate (FPS)**
- **Inference latencies** (GPU & CPU times)

## Repository Structure
```
├── ultrahelper/
│   ├── load.py              # load_hardware_model & load_postprocessor
│   ├── nn/pose.py           # ModifiedPoseHead & ModifiedPosePostprocessor
│   └── utils/monitor_4d.py  # 4D‑tensor monitoring hooks (optional)
├── pipeline.py              # ParallelInferencePipeline implementation
├── IMG_5692.mov             # Example input video
├── outputs/                 # Directory for output video
│   └── output.mp4           # Processed video
└── README.md                # This file
```

## Dependencies
- Python 3.8+
- PyTorch
- OpenCV (`cv2`)
- Ultralytics YOLOv8
- `ultrahelper` package (local)

Install with pip:
```bash
pip install torch torchvision opencv-python ultralytics
```  
(Plus ensure `ultrahelper/` is on your `PYTHONPATH`.)

## Usage
1. **Prepare your models**
   - Define your deployment (hardware) model in `ultrahelper.load.load_hardware_model()`.
   - Define your CPU postprocessor in `ultrahelper.load.load_postprocessor()`.

2. **Run the pipeline**
   ```bash
   python pipeline.py
   ```
   This will:
   - Spawn two worker processes:
     - **GPU worker** (_hardware_worker_): loads frames, resizes to 640×640, runs the hardware model, and pushes features to a shared queue.
     - **CPU worker** (_postprocessor_worker_): takes features, decodes detections & keypoints, draws annotations, and measures CPU latency.
   - Main process reads from the CPU worker’s output queue, writes annotated frames to `outputs/output.mp4`, and displays them in a window.

3. **Controls**
   - Press **q** in the display window to exit gracefully.
   - The pipeline will also print and overlay on each frame:
     - `FPS: ...`
     - `GPU: ...ms`
     - `CPU: ...ms`

## API Details
- **`load_hardware_model()`**: Returns a PyTorch `nn.Module` with only the convolutional head (`ModifiedPoseHead`).
- **`load_postprocessor()`**: Returns a postprocessor module (`ModifiedPosePostprocessor`) that takes raw feature tensors and returns:
  - Detection boxes
  - Raw keypoint heatmaps
  - Decoded keypoints

- **`decode_keypoint_heatmaps_multi`**: Utility to convert heatmap outputs to (x, y, confidence) for each keypoint.
- **`draw_keypoints` & `draw_connections`**: OpenCV helpers to visualize keypoints and skeleton edges.

## Extending & Customizing
- **Video source**: Modify `SOURCE` in `pipeline.py` to use webcam (`0`) or other videos.
- **Model paths**: Point to your custom `.pt` or YAML-defined models by editing `ultrahelper/load.py`.
- **Thresholds**: Adjust NMS and confidence thresholds in `_postprocessor_worker`.

## License
This project is released under the AGPL‑3.0 License (same as Ultralytics). Refer to [LICENSE](https://ultralytics.com/license) for details.

---
*Authored by Mahsa Raeisinezhad.*

