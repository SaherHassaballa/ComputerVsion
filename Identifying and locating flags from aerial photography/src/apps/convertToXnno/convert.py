# export_ultralytics_fix.py
from ultralytics import YOLO
import glob
import os
import shutil
import time

pt_path = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\fuck any one8\weights\best.pt"
desired_onnx = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\fuck any one8\weights\best.onnx"

print("Loading model from:", pt_path)
model = YOLO(pt_path)

print("Exporting to ONNX (this may take a while)...")
# نفّذ التصدير — بدون 'file' لأن الـ API لا يقبلها كما CLI
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=608)
print("Export finished (ultralytics saved the ONNX file somewhere).")

# نبحث عن أحدث ملف .onnx في مجلد المشروع (أو في current dir)
# (Ultralytics عادة يحفظ الملف بالقرب من weights أو في project folder)
search_dirs = [
    os.getcwd(),
    os.path.dirname(pt_path),
    os.path.join(os.path.dirname(pt_path), "weights"),
]

candidates = []
for d in search_dirs:
    if not os.path.isdir(d):
        continue
    candidates += glob.glob(os.path.join(d, "*.onnx"))

# إذا لم نجد ملفات، نبحث أعمق (عادة لا يحتاج)
if not candidates:
    for root, _, files in os.walk(os.getcwd()):
        for f in files:
            if f.endswith(".onnx"):
                candidates.append(os.path.join(root, f))

if not candidates:
    print("⚠️ لم أجد ملف .onnx تلقائيًا. ابحث يدوياً في مجلد المشروع أو شاركني مخرجات التيرمنال.")
else:
    # نأخذ أحدث ملف حسب وقت التعديل
    newest = max(candidates, key=os.path.getmtime)
    print("Found ONNX file:", newest)
    # انقله للمسار اللي انت عايزه (overwrite لو موجود)
    try:
        shutil.move(newest, desired_onnx)
        print("Moved ONNX to:", desired_onnx)
    except Exception as e:
        print("Could not move file automatically:", e)
        print("You can manually move", newest, "to", desired_onnx)
