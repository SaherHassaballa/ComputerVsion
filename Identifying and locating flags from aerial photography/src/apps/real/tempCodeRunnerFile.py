
import threading
import time
import cv2
import numpy as np
from collections import deque
import asyncio
import math

# ultralytics (YOLO .pt)
from ultralytics import YOLO

# MAVSDK
from mavsdk import System

# geographiclib
from geographiclib.geodesic import Geodesic