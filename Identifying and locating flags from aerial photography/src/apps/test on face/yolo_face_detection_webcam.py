#!/usr/bin/env python3
"""
YOLOv8 Face Detection + Optional MAVSDK Telemetry
-------------------------------------------------
Requirements:
    pip install ultralytics opencv-python mavsdk geographiclib

Press 'q' to quit.
"""

import cv2
import threading
import asyncio
import math
import numpy as np
from ultralytics import YOLO
from geographiclib.geodesic import Geodesic
from mavsdk import System

# ---------------- CONFIG ---------------- #

MODEL_PATH = r"C:\Users\saher\.cache\huggingface\hub\models--arnabdhar--YOLOv8-Face-Detection\snapshots\52fa54977207fa4f021de949b515fb19dcab4488\model.pt"

CAMERA_INDEX = 0
MAVLINK_CONN = "udpin://192.168.43.135:14550"

K = np.array([[1500,0,960],
              [0,1500,540],
              [0,0,1]],dtype=np.float64)

DIST = np.zeros((1,5))
GROUND_ALT = 0.0
GEOD = Geodesic.WGS84

# ---------------------------------------- #

try:
    from telemetry import Telemetry
    telemetry = Telemetry()
except Exception:
    class Dummy:
        def __init__(self):
            self.lock = threading.Lock()
            self.is_ready=False
            self.lat=self.lon=self.alt=0
            self.roll=self.pitch=self.yaw=0
    telemetry=Dummy()

def rotation_matrix(r,p,y):
    cr,sr=np.cos(r),np.sin(r)
    cp,sp=np.cos(p),np.sin(p)
    cy,sy=np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz@Ry@Rx

def pixel_to_geo(cx,cy):
    with telemetry.lock:
        if not telemetry.is_ready:
            return None
        lat,lon,alt=telemetry.lat,telemetry.lon,telemetry.alt
        roll,pitch,yaw=telemetry.roll,telemetry.pitch,telemetry.yaw

    pts=np.array([[[float(cx),float(cy)]]],dtype=np.float32)
    und=cv2.undistortPoints(pts,K,DIST)
    ray=np.array([[und[0,0,0]],[und[0,0,1]],[1.0]])
    ray=rotation_matrix(roll,pitch,yaw)@ray

    ez=float(ray[2,0])
    if abs(ez)<1e-6:
        return None

    scale=(alt-GROUND_ALT)/ez
    if scale<0:
        return None

    east=float(ray[0,0]*scale)
    north=float(ray[1,0]*scale)

    d=math.hypot(east,north)
    if d==0:
        return lat,lon

    az=math.degrees(math.atan2(east,north))
    g=GEOD.Direct(lat,lon,az,d)
    return g["lat2"],g["lon2"]

async def mav_task():
    drone=System()
    try:
        await drone.connect(system_address=MAVLINK_CONN)
    except Exception:
        return

    async def pos():
        async for p in drone.telemetry.position():
            with telemetry.lock:
                telemetry.lat=p.latitude_deg
                telemetry.lon=p.longitude_deg
                telemetry.alt=getattr(p,"absolute_altitude_m",0) or p.relative_altitude_m
                telemetry.is_ready=True

    async def att():
        async for a in drone.telemetry.attitude():
            with telemetry.lock:
                telemetry.roll=a.roll_rad
                telemetry.pitch=a.pitch_rad
                telemetry.yaw=a.yaw_rad

    await asyncio.gather(pos(),att())

def start_mav():
    try:
        asyncio.run(mav_task())
    except Exception:
        pass

def main():

    threading.Thread(target=start_mav,daemon=True).start()

    model=YOLO(MODEL_PATH)

    cap=cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    while True:

        ok,frame=cap.read()

        if not ok:
            break

        results=model(frame,conf=0.35,verbose=False)

        annotated=frame.copy()

        if len(results)>0 and results[0].boxes is not None:

            for box in results[0].boxes:

                x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())

                cx=(x1+x2)//2
                cy=(y1+y2)//2

                geo=pixel_to_geo(cx,cy)

                if geo:
                    txt=f"{geo[0]:.6f}, {geo[1]:.6f}"
                else:
                    txt="Unknown"

                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(annotated,(cx,cy),4,(0,0,255),-1)

                cv2.putText(
                    annotated,
                    "Face",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Center: ({cx}, {cy})",
                    (x1,y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,0),
                    1
                )

                cv2.putText(
                    annotated,
                    txt,
                    (x1,y2+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    1
                )

        cv2.imshow("YOLO Face Detection",annotated)

        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
