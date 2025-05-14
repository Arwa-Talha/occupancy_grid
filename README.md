# occupancy_grid
# Autonomous Driving Perception & Mapping with CARLA, YOLOP, Depth & LiDAR

This repository provides an integrated real-time perception and mapping pipeline for autonomous driving, using the CARLA simulator. It combines semantic segmentation, obstacle detection, and BEV (Bird's Eye View) mapping using YOLOPv2, Depth Camera, and 2D LiDAR.

---

## 🚗 Project Overview

We use the CARLA simulator and a combination of perception modules to:

* Detect lanes and drivable areas using YOLOPv2
* Extract obstacle geometry using Depth and LiDAR sensors
* Project all outputs to a unified BEV cost map
* Prepare the fused BEV map for downstream path planning

---

## 📦 Components

### Sensors (CARLA)

| Sensor       | Purpose                                   |
| ------------ | ----------------------------------------- |
| RGB Camera   | Lane & road segmentation via YOLOP        |
| Depth Camera | 3D distance estimation for BEV projection |
| 2D LiDAR     | Accurate obstacle mapping                 |

### Modules

| Module           | Function                                    |
| ---------------- | ------------------------------------------- |
| YOLOPv2          | Detect lanes & drivable areas               |
| Depth Projection | Convert depth image into obstacle points    |
| LiDAR Projection | Filter and grid LiDAR points                |
| BEV Fusion       | Combine all layers into a semantic cost map |

---

## 📍 BEV Cost Map Values

| Grid Value | Meaning                 |
| ---------- | ----------------------- |
| 0          | Free space              |
| 1          | Lane lines              |
| 2          | Drivable road           |
| 3          | Obstacles (Depth/LiDAR) |
| 50         | Unknown                 |
| 100        | Hard Obstacle           |

---

## 🧠 Pipeline Flow

1. **RGB Camera** → YOLOPv2 → Segmentation masks (lane, road)
2. **Depth Camera** → Project depth to 3D → Occupancy grid
3. **LiDAR Sensor** → Point cloud → BEV grid overlay
4. **Fusion** → Combine all into final BEV map
5. **Output** → Video + BEV map for planner

---

## 🔧 Requirements

* Python 3.7
* CARLA 0.9.15
* PyTorch
* OpenCV
* NumPy

Install requirements:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

### 1. Start CARLA Simulator

```bash
./CarlaUE4.sh -quality-level=Epic
```

Or on Windows, launch `CarlaUE4.exe`

### 2. Run the main script

```bash
python Carla.py --model-path path/to/yolopv2.pt --img-size 640
```

YOLOPv2 will process RGB frames, and all sensor data will be fused into BEV.

---

## 📁 Output

* `output_X.mp4` : Inference video
* `bev_map_X.png` : BEV map visualization
* `debug_*.png` : Intermediate segmentation/depth outputs

---

## 📌 Notes

* Ensure the model path points to a valid TorchScript YOLOPv2 `.pt` file
* Make sure depth decoding matches CARLA format
* Use consistent image sizes during preprocessing (e.g., 640x360)
