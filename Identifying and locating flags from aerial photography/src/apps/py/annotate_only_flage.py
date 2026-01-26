from ultralytics import YOLO
import os , cv2


path_images = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\images"
label_path = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\labels"

os.makedirs(path_images, exist_ok=True)
os.makedirs(label_path, exist_ok=True)


model = YOLO(
    r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\Allah_flag_not\weights\best.pt"
)

for num, img in enumerate(os.listdir(path_images)):

    image_path = os.path.join(path_images, img)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")  # تعديل: تحقق من قراءة الصورة
        continue

    img_width, img_height = (
        image.shape[1],
        image.shape[0],
    )  # تعديل: حذف التكرار واستخدام مرة واحدة

    res = model.predict(source=image_path, imgsz=960, conf=0.3, verbose=True)[0]

    file_name = os.path.join(label_path, f"{img.split('.')[0]}.txt")
    with open(file_name, "w") as file:
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                box_w = x2 - x1
                box_h = y2 - y1
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_w_norm = box_w / img_width
                box_h_norm = box_h / img_height

                file.write(
                    f"{0} {x_center_norm} {y_center_norm} {box_w_norm} {box_h_norm}\n"
                )# تعديل: تنبيه في حال عدم وجود التصنيف

    print(f"image num : {num} done ")

