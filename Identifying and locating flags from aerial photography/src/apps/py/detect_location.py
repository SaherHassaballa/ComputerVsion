import cv2
import numpy as np
from ultralytics import YOLO
from geopy.distance import geodesic
import threading
import queue

# Define video and model paths
VIDEO_PATH = "rtsp://192.168.144.25:8554/main.264"
MODEL_PATH = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\fuck any one8\weights\best.pt"

# Camera intrinsic matrix and distortion coefficients
K = np.array([[10600, 0, 1920], [0, 10600, 1080], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5,), dtype=np.float32)

# Drone GPS location and altitude (in meters)
DRONE_LAT, DRONE_LON = 30.1, 30.6
DRONE_ALT = 70.0

# Load YOLO model
model = YOLO(MODEL_PATH)

# frame queue
frame_queue = queue.Queue(maxsize=5)


# Convert pixel coordinates to a 3D ray in camera space
def undistort_and_ray(cx: int, cy: int):
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, K, DIST_COEFFS)
    x, y = norm[0, 0]
    return np.array([x, y, 1.0], dtype=np.float32)


# Convert a pixel center point to geographic coordinates (latitude, longitude)
def pixel_to_geo(cx: int, cy: int):
    ray = undistort_and_ray(cx, cy)
    scale = DRONE_ALT / ray[2]
    dx, dy = ray[0] * scale, ray[1] * scale
    mid = geodesic(meters=dy).destination((DRONE_LAT, DRONE_LON), 0)  # N-S
    dest = geodesic(meters=dx).destination((mid.latitude, mid.longitude), 90)  # E-W
    return dest.latitude, dest.longitude


# Thread for video capture
def video_reader(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()


# Thread for detection and display
def video_processor():
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", 900, 520)

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Resize frame (to speed up)
            resized = cv2.resize(frame, (640, 360))

            # Run YOLO detection
            results = model.predict(resized, conf=0.4, imgsz=640, verbose=False)

            for res in results:
                boxes = res.boxes.data.cpu().numpy()
                for x1, y1, x2, y2, conf, cls_id in boxes:
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    label = model.names[int(cls_id)]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    lat, lon = pixel_to_geo(cx, cy)

                    # Draw box + label
                    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(resized, f"{lat:.5f}, {lon:.5f}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Detection", resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    t1 = threading.Thread(target=video_reader, args=(VIDEO_PATH,))
    t2 = threading.Thread(target=video_processor)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
