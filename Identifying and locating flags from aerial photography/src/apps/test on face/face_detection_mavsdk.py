#!/usr/bin/env python3
# detection_pt_gpu.py
# Requires: ultralytics, mavsdk, opencv-python, numpy, geographiclib

import threading
import time
import cv2
import numpy as np
from collections import deque
import asyncio
import math
from telemetry import *
import os

# ultralytics (YOLO .pt)

# MAVSDK
from mavsdk import System

# geographiclib
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84


# Face detector (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")


# ---------- CONFIG ----------
VIDEO_PATH = r"D:\Saher Hassaballah\Downloads\Telegram Desktop\REC_0005.mp4"

MODEL_PT = r"d:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\fuck any one8\weights\best.pt"
MAVLINK_CONN = "udpin://192.168.43.135:14550"
frameNum = 0
coord_file = r"C:\Users\saher\Desktop\Github Projects\computer vision\projects\Identifying and locating flags from aerial photography\src\data\prediction\coord"
# Camera intrinsics
K = np.array(
    [[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
distCoeffs = np.zeros((1, 5), dtype=np.float64)

# Camera offset relative to drone body (meters)
cam_trans_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
cam_rot_offset = np.eye(3, dtype=np.float64)

# Ground altitude reference
ground_alt = 0.0
# ---------------------------

# thread-safe frame queue
frameQueue = deque()
qMutex = threading.Lock()
qCond = threading.Condition(qMutex)
MAX_QUEUE = 5
running = threading.Event()
running.set()

# telemetry shared structure


telemetry = Telemetry()


# helper: rotation matrix from roll,pitch,yaw (body->ENU)
def rotation_matrix_from_euler(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return R_z @ R_y @ R_x


# pixel -> geo conversion (Geodesic fallback)
def pixel_to_geo(cx, cy):
    with telemetry.lock:
        if not telemetry.is_ready:
            return None
        lat = telemetry.lat
        lon = telemetry.lon
        alt = telemetry.alt
        roll = telemetry.roll
        pitch = telemetry.pitch
        yaw = telemetry.yaw
        print("alt ------>>>>>>>>>> ", alt)
        print("alt ------>>>>>>>>>> ", lon)
        print("alt ------>>>>>>>>>> ", lat)

    # undistort -> normalized coords
    pts = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, distCoeffs)
    x = float(und[0, 0, 0])
    y = float(und[0, 0, 1])
    ray_cam = np.array([x, y, 1.0], dtype=np.float64).reshape((3, 1))

    ray_body = cam_rot_offset @ ray_cam
    R = rotation_matrix_from_euler(roll, pitch, yaw)
    ray_enu = R @ ray_body

    cam_alt = alt + cam_trans_offset[2]
    ez = float(ray_enu[2, 0])
    if abs(ez) < 1e-8:
        return None
    scale = (cam_alt - ground_alt) / ez
    print("scaled ----->>>>>>>>", scale)
    if scale < 0:
        return None

    east = float(ray_enu[0, 0] * scale)
    north = float(ray_enu[1, 0] * scale)

    distance = math.hypot(east, north)
    if distance == 0.0:
        return (lat, lon)
    azimuth_deg = math.degrees(math.atan2(east, north))
    dest = geod.Direct(lat, lon, azimuth_deg, distance)
    return (dest["lat2"], dest["lon2"])


# video reader thread
def reader_thread(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Reader] Failed to open stream: {source}")
        running.clear()
        return
    while running.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.3)
            continue
        with qCond:
            if len(frameQueue) >= MAX_QUEUE:
                frameQueue.popleft()
            frameQueue.append(frame)
            qCond.notify()
    cap.release()


# MAVSDK thread
def mavsdk_thread(connection_url):
    async def run_mavsdk():
        drone = System()
        print(f"[MAV] connecting to: {connection_url}")
        try:
            await drone.connect(system_address=connection_url)
        except Exception as e:
            print(f"[MAV] Connection error: {e}")
            running.clear()
            return

        print("[MAV] waiting for system connection...")
        try:
            async for state in drone.core.connection_state():
                if state.is_connected:
                    print("[MAV] System connected.")
                    break
                await asyncio.sleep(0.1)
        except Exception:
            await asyncio.sleep(1.0)

        async def attitude_task():
            async for att in drone.telemetry.attitude():
                with telemetry.lock:
                    telemetry.roll = att.roll_rad
                    telemetry.pitch = att.pitch_rad
                    telemetry.yaw = att.yaw_rad
                if not running.is_set():
                    break

        async def position_task():
            async for pos in drone.telemetry.position():
                with telemetry.lock:
                    telemetry.lat = pos.latitude_deg
                    telemetry.lon = pos.longitude_deg
                    a = getattr(pos, "absolute_altitude_m", None)
                    if a is None or a == 0.0:
                        telemetry.alt = pos.relative_altitude_m
                        print(
                            "telemetry.alt --------------->>>>>>>>>>>> ", telemetry.alt
                        )
                        print(
                            "telemetry.lat --------------->>>>>>>>>>>> ", telemetry.lat
                        )
                        print(
                            "telemetry.lon --------------->>>>>>>>>>>> ", telemetry.lon
                        )
                    else:
                        telemetry.alt = a
                        print("telemetry.alt --------------->>>>>>>>>>>> ", a)
                    if not telemetry.is_ready:
                        telemetry.is_ready = True
                        print("[MAV] Telemetry READY.")
                if not running.is_set():
                    break

        tasks = [
            asyncio.create_task(attitude_task()),
            asyncio.create_task(position_task()),
        ]
        try:
            while running.is_set():
                await asyncio.sleep(0.2)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.sleep(0.1)
            print("[MAV] telemetry tasks stopped.")

    try:
        asyncio.run(run_mavsdk())
    except Exception as e:
        print(f"[MAV] Exception: {e}")
        running.clear()


# processing thread
def processor_thread():

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    while running.is_set():

        frame = None

        with qCond:
            if not qCond.wait_for(
                lambda: len(frameQueue) > 0 or not running.is_set(),
                timeout=0.1,
            ):
                continue

            if len(frameQueue) > 0:
                frame = frameQueue.pop()
                frameQueue.clear()

        if frame is None:
            continue

        frame_h, frame_w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        for (x, y, w, h) in faces:

            cx = x + w // 2
            cy = y + h // 2

            geo = pixel_to_geo(cx, cy)

            if geo is not None:
                lat, lon = geo
                coord_txt = f"{lat:.6f}, {lon:.6f}"
            else:
                coord_txt = "No Data"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

            cv2.putText(frame, "Face", (x, max(30,y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame, coord_txt,
                        (x, min(frame_h-5, y+h+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,255), 1)

            print("--------------------------------")
            print(f"Center Pixel: ({cx}, {cy})")
            print(coord_txt)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running.clear()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 1) Start MAVSDK thread first (background)
    print("[Main] Starting MAVSDK thread...")
    t_mav = threading.Thread(target=mavsdk_thread, args=(MAVLINK_CONN,), daemon=True)
    t_mav.start()

    # 2) Start a stopper thread that waits for Enter and clears `running`
    def wait_enter_and_stop():
        input("\nPress Enter anytime to stop the program...\n")
        print("[Main] Enter pressed -> stopping...")
        running.clear()

    t_stopper = threading.Thread(target=wait_enter_and_stop, daemon=True)
    t_stopper.start()

    # 3) Wait for telemetry to become ready (MAV connection) with timeout
    print("[Main] Waiting for MAV telemetry to become READY (timeout 20s)...")
    mav_timeout = 20.0  # seconds, change if you want longer
    t0 = time.time()
    while not telemetry.is_ready and running.is_set():
        if time.time() - t0 > mav_timeout:
            print("[Main] Timeout waiting for MAV telemetry. Aborting start.")
            running.clear()
            break
        print("[Main] still waiting for telemetry...")  # progress message
        time.sleep(0.5)

    if not running.is_set():
        print(
            "[Main] Not starting video/processing because running flag cleared or timeout."
        )
        # ensure MAV thread can exit
        t_mav.join(timeout=2.0)
    else:
        print("[Main] MAV telemetry READY. Checking video source...")

        # 4) Check video source / camera can be opened BEFORE starting reader
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[Main] ERROR: Cannot open video source: {VIDEO_PATH}")
            cap.release()
            running.clear()
            t_mav.join(timeout=2.0)
        else:
            print(
                "[Main] Video source opened OK. Starting reader and processor threads..."
            )
            cap.release()

            # Start reader + processor threads (non-daemon so they can cleanup)
            t_reader = threading.Thread(
                target=reader_thread, args=(VIDEO_PATH,), daemon=False
            )
            t_proc = threading.Thread(target=processor_thread, daemon=False)

            t_reader.start()
            t_proc.start()

            # 5) Main loop: wait until running cleared (by Enter, error, or Ctrl+C)
            try:
                while running.is_set():
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\n[Main] KeyboardInterrupt -> stopping...")
                running.clear()
            finally:
                print("[Main] Shutting down threads...")
                # Ask threads to stop, then join (give them a short time)
                t_reader.join(timeout=3.0)
                t_proc.join(timeout=3.0)
                t_mav.join(timeout=3.0)
                print("[Main] Exited cleanly.")
