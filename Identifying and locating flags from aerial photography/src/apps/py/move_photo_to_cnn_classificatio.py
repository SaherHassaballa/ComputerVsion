import os
from pathlib import Path
import shutil

# المسارات
SRC_DIR = Path(r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\flag_classification\images")
DEST_DIR = Path(r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\flag_classification_lite")

# إنشاء مجلد الوجهة لو مش موجود
DEST_DIR.mkdir(parents=True, exist_ok=True)

# نقل الصور حسب اسم الدولة
for img_path in SRC_DIR.iterdir():
    if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        parts = img_path.stem.split('_')
        if not parts:
            continue
        country = parts[0].lower()

        # إنشاء مجلد للدولة
        country_folder = DEST_DIR / country
        country_folder.mkdir(parents=True, exist_ok=True)

        # تحديد المسار الجديد
        new_path = country_folder / img_path.name

        # نسخ أو نقل الصورة
        shutil.copy(img_path, new_path)  # استخدم shutil.move لو عاوز تنقل بدل نسخ

        print(f"✅ Moved {img_path.name} to {country_folder}")
