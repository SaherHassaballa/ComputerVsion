from unittest import result
import cv2
from ultralytics import YOLO

VIDEO_PATH  = r"D:\Saher Hassaballah\Downloads\Telegram Desktop\REC_0005.mp4"
MODEL_PATH = r" D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\fuck any one6\weights\best.pt"

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 900, 520)
ret = True
model = YOLO(MODEL_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 4000)
while ret :
    ret , frame = cap.read()
    if not ret :
        break
    results = model.predict(frame , conf = 0.3, imgsz = 544)
    for result in results :
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1 , y1 , x2 , y2 = map(int , box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"{label} ({conf:.2f})", (cx, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()