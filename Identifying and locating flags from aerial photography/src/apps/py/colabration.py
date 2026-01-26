import cv2
import numpy as np
import glob
import os

# إعدادات لوحة الشطرنج
chessboard_size = (5, 7)  # لأن عندك لوحة 8x6 مربعات
square_size = 1.0         # ممكن بالسم أو بالمتر

# إعداد نقاط 3D في العالم الحقيقي
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# المتغيرات اللي هنخزن فيها البيانات
objpoints = []
imgpoints = []
# قراءة الصور
image_folder = 'C:/Users/saher/Desktop/workshop/Computer_cv/camera calobration/calibration_images/*.jpg'
images = glob.glob(image_folder)

print(f"🔍 Found {len(images)} images.")

# التأكد إن فيه صور
if len(images) == 0:
    print("⚠️ لا توجد صور في المسار المحدد!")
    exit()

# إنشاء مجلد لحفظ الصور اللي فيها الزوايا
os.makedirs("output_corners", exist_ok=True)

# المتابعة مع كل صورة
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"⚠️ فشل في قراءة الصورة: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        out_name = os.path.join("output_corners", os.path.basename(fname))
        cv2.imwrite(out_name, img)
        print(f"✅ Corners detected: {fname}")
    else:
        print(f"❌ Failed to detect corners: {fname}")

# التحقق من وجود نقاط كافية للمعايرة
if len(objpoints) == 0:
    print("❌ لم يتم اكتشاف أي زوايا. تأكد من جودة الصور.")
    exit()

# إجراء المعايرة
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# حفظ النتائج
np.savez("camera_calibration.npz", K=camera_matrix, dist=dist_coeffs)

# طباعة النتائج
print("\n✅ Camera Calibration Completed Successfully!\n")
print("🎯 Camera Matrix:\n", camera_matrix)
print("\n🔧 Distortion Coefficients:\n", dist_coeffs)
