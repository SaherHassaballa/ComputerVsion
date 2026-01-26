import logging
from pathlib import Path
import random
import numpy as np
import cv2, os 
from skimage import io
from ultralytics import YOLO
from tqdm import tqdm

# ================== Hardcoded Paths ==================
INPUT_DIR = Path(r"D:/github lite/cv_projects/Identifying and locating flags from aerial photography/data/inserted_flags_in_background/images")
OUTPUT_DIR = Path(r"D:/github lite/cv_projects/Identifying and locating flags from aerial photography/data/flag_classification/images")
OUTPUT_DIR_LABELS = Path(r"D:/github lite/cv_projects/Identifying and locating flags from aerial photography/data/flag_classification/labels")
MODEL_PATH = Path(r"D:/github lite/cv_projects/Identifying and locating flags from aerial photography/models/trained_models/flags_final_yolov8x_new_v1_flag_or_not_are_renamed/weights/best.pt")
# =====================================================

# Parameters
MARGIN = 0
MIN_CROP_SIZE = 20  # الحد الأدنى لحجم العلم المقتص
ACCEPTED_SUFFIXES = {'.jpg', '.jpeg', '.png'}

data = [
    {"name": "algeria"}, {"name": "argentina"}, {"name": "brazil"},
    {"name": "china"},   {"name": "egypt"},     {"name": "ethiopia"}, {"name": "france"},
    {"name": "germany"}, {"name": "india"},     {"name": "iran"},     {"name": "iraq"},
    {"name": "italy"},   {"name": "japan"},     {"name": "korea"},    {"name": "lebanon"},
    {"name": "libya"},   {"name": "morocco"},   {"name": "pakistan"},{"name": "portugal"},
    {"name": "russian"}, {"name": "saudi arabia"},{"name": "spain"},   {"name": "sudan"},
    {"name": "sweden"},  {"name": "switzerland"},{"name": "tunisia"},{"name": "turkey"},
]

def get_labels(data):
    names = [c['name'] for c in data]
    labels_dictionary = {name: idx for idx, name in enumerate(names)}
    return names, labels_dictionary

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def process_image(img_path: Path, model: YOLO, labels_dictionary: dict):
    results = model.predict(str(img_path), conf=0.6, verbose=False)
    if not results:
        logging.debug(f"No results list for {img_path.name}")
        return

    res = results[0]
    if res.boxes is None or res.boxes.xyxy.shape[0] == 0:
        logging.debug(f"No detections for {img_path.name}")
        return

    img = io.imread(str(img_path))
    h, w = img.shape[:2]

    # استخراج اسم البلد بأمان
    parts = img_path.stem.split('_')
    country = parts[0].lower() if parts else "unknown"

    # إنشاء مجلدات الحفظ منفصلة لكل بلد
    save_img_dir = OUTPUT_DIR
    save_lbl_dir = OUTPUT_DIR_LABELS
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_lbl_dir.mkdir(parents=True, exist_ok=True)

    boxes = res.boxes.xyxy.cpu().numpy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # تطبيق المارجن
        x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
        x2, y2 = min(w, x2 + MARGIN), min(h, y2 + MARGIN)

        crop = img[y1:y2, x1:x2]
        # التحقق من أن القص ليس فارغاً أو صغيرًا جدًا
        if crop.size == 0:
            logging.warning(f"🚫 Empty crop, skipping: {img_path.name}")
            continue
        if crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE:
            logging.warning(f"⚠️ Crop too small, skipping: {img_path.name}")
            continue

        # حفظ الصورة
        save_name = f"{country}_{img_path.stem}_{idx}.jpg"
        save_path = save_img_dir / save_name.lower()
        io.imsave(str(save_path), crop)
        logging.info(f"✅ Saved: {save_path}")

        # # حساب إحداثيات YOLO normalized
        # x_center = crop.shape[1]//2
        # y_center = crop.shape[0]//2
        # box_w = crop.shape[1] / crop.shape[1]
        # box_h = crop.shape[0] / crop.shape[0]

        # كتابة الليبل
        label_idx = labels_dictionary.get(country)
        label_path = save_lbl_dir / f"{country}_{img_path.stem}_{idx}.txt"
        with open(label_path, 'w') as f:
            if label_idx is not None:
                f.write(f"{label_idx} 0.5 0.5 1 1\n")
            else:
                logging.warning(f"⚠️ Country label not found for '{country}'")
def main():
    setup_logging()
    names, labels_dictionary = get_labels(data)
    model = YOLO(str(MODEL_PATH))

    images = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES]
    for img_path in tqdm(images, desc="📷 Processing images"):
        try:
            process_image(img_path, model, labels_dictionary)
        except Exception as e:
            logging.error(f"❌ Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    main()
