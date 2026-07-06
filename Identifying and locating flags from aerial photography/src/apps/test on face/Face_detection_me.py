# import cv2

# # sys.exit() is the standard way to terminate a Python script and is preferred in production code.
# # exit() ends the program so it doesn't continue and cause more errors.

# # Load the Haar Cascade classifier
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# # Check if the classifier loaded correctly
# if face_cascade.empty():
#     print("Error: Could not load face detector.")
#     exit() 

# # Open the default webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam opened successfully
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# print("Press 'q' to quit.")

# while True:
#     # Read a frame
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to grab frame.")
#         break

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(40, 40)
#     )

#     # Draw rectangles and labels
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(
#             frame,
#             "Face",
#             (x, y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )

#     # Display the number of detected faces
#     cv2.putText(
#         frame,
#         f"Faces: {len(faces)}",
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255, 0, 0),
#         2
#     )

#     # Show the output
#     cv2.imshow("Face Detection", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#******************************************************************************************************#

#!/usr/bin/env python3
"""
Face detection + optional MAVSDK telemetry.
- Webcam (camera index 0)
- Face detection using OpenCV Haar Cascade
- If telemetry is available -> GPS coordinates shown
- Otherwise -> "Unknown"
"""

import cv2
import threading
import time
import numpy as np
import math
import asyncio
from collections import deque
from geographiclib.geodesic import Geodesic
from mavsdk import System

try:
    from telemetry import *
    telemetry = Telemetry()
except Exception:
    class Dummy:
        def __init__(self):
            self.lock = threading.Lock()
            self.is_ready=False
            self.lat=0
            self.lon=0
            self.alt=0
            self.roll=0
            self.pitch=0
            self.yaw=0
    telemetry=Dummy()

MAVLINK_CONN="udpin://192.168.43.135:14550"

K=np.array([[1500,0,960],[0,1500,540],[0,0,1]],dtype=np.float64)
distCoeffs=np.zeros((1,5),dtype=np.float64)
ground_alt=0.0
geod=Geodesic.WGS84

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

frameQueue=deque(maxlen=5)
qCond=threading.Condition()
running=True

def rotation_matrix_from_euler(r,p,y):
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
    und=cv2.undistortPoints(pts,K,distCoeffs)
    x=float(und[0,0,0]); y=float(und[0,0,1])
    ray=np.array([[x],[y],[1.0]])
    R=rotation_matrix_from_euler(roll,pitch,yaw)
    ray=R@ray
    ez=float(ray[2,0])
    if abs(ez)<1e-6:
        return None
    scale=(alt-ground_alt)/ez
    if scale<0:
        return None
    east=float(ray[0,0]*scale)
    north=float(ray[1,0]*scale)
    d=math.hypot(east,north)
    if d==0:
        return (lat,lon)
    az=math.degrees(math.atan2(east,north))
    dest=geod.Direct(lat,lon,az,d)
    return dest["lat2"],dest["lon2"]

def reader():
    global running
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while running:
        ok,frame=cap.read()
        if ok:
            with qCond:
                frameQueue.append(frame)
                qCond.notify()
    cap.release()

async def mav():
    drone=System()
    try:
        await drone.connect(system_address=MAVLINK_CONN)
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

def mav_thread():
    try:
        asyncio.run(mav())
    except:
        pass

def processor():
    global running
    while running:
        with qCond:
            if not frameQueue:
                qCond.wait(timeout=0.1)
                continue
            frame=frameQueue.pop()
            frameQueue.clear()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.1,5,minSize=(40,40))
        for (x,y,w,h) in faces:
            cx=x+w//2
            cy=y+h//2
            geo=pixel_to_geo(cx,cy)
            txt=f"{geo[0]:.6f},{geo[1]:.6f}" if geo else "Unknown"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.putText(frame,"Face",(x,y-8),0,0.6,(0,255,0),2)
            cv2.putText(frame,f"Center:({cx},{cy})",(x,y+h+18),0,0.5,(255,255,0),1)
            cv2.putText(frame,txt,(x,y+h+38),0,0.5,(0,255,255),1)
        cv2.imshow("Face Detection",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            running=False
    cv2.destroyAllWindows()

if __name__=="__main__":
    threading.Thread(target=mav_thread,daemon=True).start()
    threading.Thread(target=reader,daemon=True).start()
    processor()
