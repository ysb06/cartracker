# SCK Cartracker

A car tracking system that processes ROS bag files from multi-camera setups to perform object detection and tracking on panoramic images.

## Overview

SCK Cartracker is designed to work with ROS bag files containing synchronized data from 4 cameras. The system stitches images from multiple cameras into panoramic views and performs object detection using the RF-DETR model.

## Features

- **Multi-camera ROS bag processing**: Extract and synchronize data from 4 camera topics
- **Image stitching**: Create panoramic views using OpenCV's Stitcher with automatic fallback
- **Object detection**: RF-DETR base model with COCO classes for vehicle and object detection
- **Real-time visualization**: Live display of detection results with bounding boxes and confidence scores

## Installation

### Prerequisites

- Python 3.11
- PDM (Python Dependency Manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd sck-cartracker
```

2. Install dependencies using PDM:
```bash
pdm install
```

### Dependencies

- **opencv-python** (≥4.12.0.88): Image processing and stitching
- **bagpy** (≥0.5): ROS bag file processing
- **tqdm** (≥4.67.1): Progress bars for data extraction
- **rfdetr** (≥1.2.1): Object detection model

## Usage

### Basic Usage

1. Place your ROS bag file in the `data/` directory
2. Run the application:

```bash
pdm run python -m scart
```

or alternatively:

```bash
python -m src.scart
```

### Data Structure

The system expects ROS bag files with the following camera topics:
- `/clpe_ros/cam_0/image_raw`
- `/clpe_ros/cam_1/image_raw`
- `/clpe_ros/cam_2/image_raw`
- `/clpe_ros/cam_3/image_raw`

### Configuration

Configuration files are located in the `config/` directory:

- `config/config.yaml`: Main configuration with Hydra defaults
- `config/dataset/songdo.yaml`: Dataset configurations
- `config/tracking/base.yaml`: Tracking parameters
- `config/model/`: Model configurations
- `config/training/base.yaml`: Training parameters

## Architecture

### Core Components

#### ScartBag (`src/scart/bagloader/v250805.py`)
- Handles ROS bag file loading and parsing
- Extracts synchronized camera data from multiple topics
- Converts ROS Image messages to numpy arrays
- Supports multiple image encodings (bgr8, rgb8, mono8, yuv422)
- Provides Korean timezone (KST) timestamp conversion

#### ObjectDetector (`src/scart/__main__.py`)
- Uses RF-DETR base model for object detection
- Optimized for inference performance
- COCO class detection with configurable confidence threshold (0.5)
- Supervision integration for annotation visualization

### Data Flow

1. **Load ROS bag**: ScartBag loads and parses ROS bag files containing multi-camera data
2. **Extract frames**: Synchronized camera frames are extracted from 4 camera topics
3. **Image stitching**: Creates panoramic views using OpenCV's Stitcher
4. **Fallback handling**: Uses center camera image if stitching fails
5. **Object detection**: Applies RF-DETR detection on both individual and stitched images
6. **Visualization**: Displays results with bounding boxes and labels using OpenCV

## Model

The system uses RF-DETR (Real-time DETR) base model for object detection:
- Pre-trained on COCO dataset
- Optimized for inference speed
- Supports 80 COCO object classes
- Model weights: `rf-detr-base.pth` (place in root directory)

## Development

### Running the Application

The main application entry point is `src/scart/__main__.py`. It processes ROS bag data sequentially, performing:

1. Bag info extraction and display
2. Camera data extraction with progress tracking
3. Image stitching for panoramic views
4. Object detection on both center camera and stitched images
5. Real-time display of annotated results

### Image Processing

- **Stitching**: Uses OpenCV's Stitcher with PANORAMA mode for optimal panoramic creation
- **Fallback**: Automatically falls back to center camera image if stitching fails
- **Encoding support**: Handles multiple ROS image encodings automatically

## License

MIT License

## Author

Seungbin Yim (ysb06@hotmail.com)