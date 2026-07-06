#!/usr/bin/env python3
"""
YOLO Face Detection + Optional GPS Coordinates
----------------------------------------------
- Webcam input
- YOLOv8 Face model
- Saves annotated detections to frame folder
- Saves coordinates to coord folder
"""

import os
import cv2
import math
import asyncio
import threading
import numpy as np
from ultralytics import YOLO
from geographiclib.geodesic import Geodesic
from mavsdk import System

MODEL_PATH = r"C:\Users\saher\.cache\huggingface\hub\models--arnabdhar--YOLOv8-Face-Detection\snapshots\52fa54977207fa4f021de949b515fb19dcab4488\model.pt"

FRAME_DIR = r"C:\Users\saher\Desktop\Github Projects\ComputerVision\Identifying and locating flags from aerial photography\src\data\prediction\frame"
COORD_DIR = r"C:\Users\saher\Desktop\Github Projects\ComputerVision\Identifying and locating flags from aerial photography\src\data\prediction\coord"
COORD_FILE = os.path.join(COORD_DIR, "coordinates.txt")

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(COORD_DIR, exist_ok=True)

CAMERA_INDEX = 0
MAVLINK = "udpin://192.168.43.135:14550"

K=np.array([[1500.,0.,960.],[0.,1500.,540.],[0.,0.,1.]])
DIST=np.zeros((1,5))
GROUND_ALT=0.0
GEOD=Geodesic.WGS84

try:
    from telemetry import Telemetry
    telemetry=Telemetry()
except:
    class T:
        def __init__(self):
            self.lock=threading.Lock()
            self.is_ready=False
            self.lat=self.lon=self.alt=0
            self.roll=self.pitch=self.yaw=0
    telemetry=T()

def R(roll,pitch,yaw):
    cr,sr=np.cos(roll),np.sin(roll)
    cp,sp=np.cos(pitch),np.sin(pitch)
    cy,sy=np.cos(yaw),np.sin(yaw)
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
    pts=np.array([[[cx,cy]]],dtype=np.float32)
    und=cv2.undistortPoints(pts,K,DIST)
    ray=np.array([[und[0,0,0]],[und[0,0,1]],[1.]])
    ray=R(roll,pitch,yaw)@ray
    ez=float(ray[2,0])
    if abs(ez)<1e-6: return None
    s=(alt-GROUND_ALT)/ez
    if s<0: return None
    east=float(ray[0,0]*s); north=float(ray[1,0]*s)
    d=math.hypot(east,north)
    if d==0: return lat,lon
    az=math.degrees(math.atan2(east,north))
    g=GEOD.Direct(lat,lon,az,d)
    return g["lat2"],g["lon2"]

async def mav():
    drone=System()
    try:
        await drone.connect(system_address=MAVLINK)
    except:
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

threading.Thread(target=lambda: asyncio.run(mav()),daemon=True).start()

model=YOLO(MODEL_PATH)
cap=cv2.VideoCapture(CAMERA_INDEX)

frame_id=0

while True:
    ok,frame=cap.read()
    if not ok:
        break

    results=model(frame,conf=0.35,verbose=False)

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
            cx=(x1+x2)//2
            cy=(y1+y2)//2

            geo=pixel_to_geo(cx,cy)
            if geo:
                coord=f"{geo[0]:.6f}, {geo[1]:.6f}"
            else:
                coord="Unknown"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.putText(frame,"Face",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.putText(frame,f"({cx},{cy})",(x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            cv2.putText(frame,coord,(x1,y2+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

            frame_name=os.path.join(FRAME_DIR,f"img_{frame_id:06d}.jpg")
            cv2.imwrite(frame_name,frame)

            with open(COORD_FILE,"a") as f:
                f.write(f"img_{frame_id:06d}.jpg | Pixel=({cx},{cy}) | GPS={coord}\n")

            frame_id+=1

    cv2.imshow("YOLO Face Detection",frame)

    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
