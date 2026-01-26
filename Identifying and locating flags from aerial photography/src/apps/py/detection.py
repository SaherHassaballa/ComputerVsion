import cv2
import numpy as np
from ultralytics import YOLO
from geopy.distance import geodesic

# ========== إعداد المسارات ========== #
IMG_PATH = r"C:\Users\saher\Pictures\Screenshots\Screenshot 2025-07-09 163409.png"
MODEL_PATH = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\flags_final_yolov8x_new_v1_flag_or_not_are_renamed\weights\best.pt"

# ========== مصفوفة الكاميرا ومعاملات ========== #
K = np.array([[2900, 0, 2048],
              [0, 2900, 1080],
              [0,    0,    1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5,), dtype=np.float32)

# ========== بيانات الطائرة ========== #
DRONE_LAT, DRONE_LON = 30.1, 30.6
DRONE_ALT = 70.0  # بالأمتار

# ========== تحميل موديل YOLO ========== #
model = YOLO(MODEL_PATH)

# ========== دوال مساعده ========== #
def undistort_and_ray(cx: int, cy: int):
    """حول نقطة بكسل إلى شعاع في فضاء الكاميرا."""
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, K, DIST_COEFFS)
    x, y = norm[0, 0]
    return np.array([x, y, 1.0], dtype=np.float32)


def pixel_to_geo(cx: int, cy: int):
    """أحسب إحداثيات جغرافية (lat, lon) لنقطة مركز المربع."""
    ray = undistort_and_ray(cx, cy)
    scale = DRONE_ALT / ray[2]
    dx, dy = ray[0] * scale, ray[1] * scale
    mid = geodesic(meters=dy).destination((DRONE_LAT, DRONE_LON), 0)
    dest = geodesic(meters=dx).destination((mid.latitude, mid.longitude), 90)
    return dest.latitude, dest.longitude


def process_image(path: str):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # إعادة تحجيم للصورة إن كانت كبيرة
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024 / max_dim
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame.copy()

    # inference مع FP16 إذا كانت GPU تدعمه
    results = model.predict(resized, conf=0.1, device=0, half=True)

    for res in results:
        boxes = res.boxes.data.cpu().numpy()
        for x1, y1, x2, y2, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = model.names[int(cls_id)]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lat, lon = pixel_to_geo(cx, cy)

            # رسم المستطيل والنص
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(resized, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            coord_text = f"{lat:.5f}, {lon:.5f}"
            cv2.putText(resized, coord_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # قصّ العلم وتكبيره (عرض فقط، بدون حفظ)
            pad = 5
            crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size > 0:
                up = cv2.resize(crop, (crop.shape[1] * 8, crop.shape[0] * 8), interpolation=cv2.INTER_LANCZOS4)
                cv2.imshow(f"Crop_{label}", up)

    # عرض النتيجة النهائية
    cv2.imshow("Detection", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image(IMG_PATH)