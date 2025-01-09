from ultralytics import YOLO

# System
import sys
import os
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Load a COCO-pretrained YOLO11n model
model = YOLO(parent_dir + "\yolo11s.pt")