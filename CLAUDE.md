# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a car tracking project (SCK Cartracker) that processes ROS bag files from multi-camera systems to perform object detection and tracking. The project is currently undergoing a major refactoring where legacy code (`cartracker/`, `cartracker_v2/`, `guiff_core/`) will be consolidated into the new `scart` package located in `src/scart/`.

## Current State

**Active Development Directory**: `src/scart/` - All new development should focus here
**Legacy Directories**: `cartracker/`, `cartracker_v2/`, `guiff_core/` - Reference only, scheduled for deletion

The project uses PDM (Python Dependency Manager) for package management with Python 3.11.

## Architecture

### Core Components

- **ScartBag** (`src/scart/bagloader/v250805.py`): Handles ROS bag file loading and camera data extraction from multiple camera topics (`/clpe_ros/cam_0/image_raw` through `/clpe_ros/cam_3/image_raw`)
- **ObjectDetector** (`src/scart/__main__.py`): Uses RF-DETR model for object detection with supervision for annotation
- **Image Stitching**: Combines multiple camera views into panoramic images using OpenCV's Stitcher

### Data Flow

1. Load ROS bag files containing multi-camera data
2. Extract synchronized camera frames from 4 camera topics  
3. Perform image stitching to create panoramic views
4. Apply RF-DETR object detection on both individual and stitched images
5. Visualize results with bounding boxes and labels

### Configuration

Configuration is managed through YAML files in the `config/` directory:
- `config/config.yaml`: Main configuration with Hydra defaults
- `config/tracking/base.yaml`: Tracking parameters
- `config/dataset/songdo.yaml`: Dataset configurations
- `config/model/`: Model configurations (YOLOvgg variants)
- `config/training/base.yaml`: Training parameters

## Development Commands

### Package Management
```bash
pdm install           # Install dependencies
pdm add <package>     # Add new dependency
pdm remove <package>  # Remove dependency
```

### Running the Application
```bash
pdm run python -m scart    # Run main application
python -m src.scart        # Alternative run method
```

### Data Processing
The application expects ROS bag files in the `data/` directory (e.g., `./data/20250801_125812_test1_0.bag`)

## Key Dependencies

- **opencv-python**: Image processing and stitching
- **bagpy**: ROS bag file processing  
- **rfdetr**: Object detection model
- **supervision**: Detection visualization and annotation
- **tqdm**: Progress bars for data extraction
- **rosbag**: ROS bag file handling

## File Locations

- **Main application**: `src/scart/__main__.py`
- **Bag loader**: `src/scart/bagloader/v250805.py`
- **Data directory**: `data/` (ROS bag files)
- **Output directory**: `outputs/` (processed videos and results)
- **Datasets**: `datasets/` (training images and configurations)
- **Model weights**: Root directory (e.g., `rf-detr-base.pth`)

## Important Notes

- The project processes 4-camera ROS bag data with Korean timezone (KST) timestamps
- Image stitching may fail for some frames; fallback uses center camera image
- Object detection uses COCO classes with RF-DETR base model
- All legacy tracking algorithms (DeepSORT, SimpleTracker) are in deprecated directories