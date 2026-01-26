# from unittest import result
# import cv2
# from ultralytics import YOLO

# VIDEO_PATH  = r"D:\Saher Hassaballah\Downloads\Telegram Desktop\REC_0005.mp4"
# MODEL_PATH = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\flags_final_yolov8x_new_v1_flag_or_not_are_renamed\weights\best.pt"

# cap = cv2.VideoCapture(VIDEO_PATH)
# cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Detection", 900, 520)
# ret = True
# model = YOLO(MODEL_PATH)

# while ret :
#     ret , frame = cap.read()
#     if not ret :
#         break
#     results = model.predict(frame , conf = 0.4 , imgsz = 1024)
#     for result in results :
#         boxes = result.boxes.cpu().numpy()
#         for box in boxes:
#             x1 , y1 , x2 , y2 = map(int , box.xyxy[0])
#             conf = box.conf[0]
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.putText(frame, f"{label} ({conf:.2f})", (cx, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.imshow("Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO

VIDEO_PATH = r"D:\Saher Hassaballah\Downloads\Telegram Desktop\REC_0005.mp4"
MODEL_PATH = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\flags_final_yolov8x_new_v1_flag_or_not_are_renamed\weights\best.pt"

cap = cv2.VideoCapture(VIDEO_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS)
model = YOLO(MODEL_PATH)

frame_pos = 0
paused = False
frame = None

cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Detection", 1200, 620)
count = 0

while cap.isOpened():
    if not paused or frame is None:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 5 == 0:  # Process every 5th frame
            results = model.predict(frame, conf=0.6, imgsz=1024, device="cpu", half=False)
            saved = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    if label == "flag":
                        saved = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if saved:
                path = rf"C:\Users\saher\Desktop\github\computer vision\projects\Identifying and locating flags from aerial photography\src\data\prediction\frame_{frame_pos}.jpg"
                cv2.imwrite(path, frame)

        count += 1

    cv2.putText(frame, f"Frame: {frame_pos}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Video Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):  # Space: pause/play
        paused = not paused
    elif key == ord('d'):  # Forward
        frame_pos += 1800
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        frame = None
    elif key == ord('a'):  # Backward
        frame_pos = max(0, frame_pos - 1800)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        frame = None
    elif not paused:
        frame_pos += 1

cap.release()
cv2.destroyAllWindows()
