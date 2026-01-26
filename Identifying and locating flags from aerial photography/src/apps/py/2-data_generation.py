import cv2
import os
import random
import numpy as np

# ===================== CONSTANTS =====================
RANDOM_SEED = 42
FOLDER_PATH = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\downloaded_flags"
EDITED_FLAGS = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background2"
BACKGROUNDS_DIR = r"C:\Users\saher\Desktop\Github Projects\computer vision\projects\Identifying and locating flags from aerial photography\src\data\back_grounds"
BACKGROUNDS_PATH = [os.path.join(BACKGROUNDS_DIR, file) for file in os.listdir(BACKGROUNDS_DIR)]

FINAL_SHAPE = [ 21 , 22 , 23 , 24 ,25 , 30, 32 ,33 , 34 , 35, 37, 39, 40]
NUM_BACKGROUNDS_TO_SAVE = 200
BRIGHTNESS_FACTORS = [1.0 , 1 , 0.9 , 0.8 , 0.7]  # خليها ثابتة للحفاظ على التفاصيل
gaussian_list = [1,1,1,3,5 , 7 , 11]  # تأثير بلور (معلق)
noises = [0, 0 ,0  , 0.1 , 0.3 , 0.5]  # تأثير نويز (معلق)
angle = [0]
# =====================================================

random.seed(RANDOM_SEED)
os.makedirs(FOLDER_PATH, exist_ok=True)
os.makedirs(EDITED_FLAGS, exist_ok=True)

flags = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]
if not flags:
    raise ValueError("❌ there aren't flags in folder")

backgrounds = [cv2.imread(path) for path in BACKGROUNDS_PATH]
backgrounds = [bg for bg in backgrounds if bg is not None]
if not backgrounds:
    raise ValueError("❌ no load for backgrounds")

height, width = backgrounds[0].shape[:2]

def load_image_with_retry(image_path, retries=3):
    for attempt in range(retries):
        img = cv2.imread(image_path)
        if img is not None:
            return img
        print(f"⏳ retry to load: {image_path} (try {attempt + 1})")
    return None

def random_flip_or_rotate(image):
    transforms = [
        lambda x: x,
        lambda x: cv2.flip(x, 0),
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.flip(x, -1),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_180),
    ]
    return random.choice(transforms)(image)

counter_for_backgrounds = 0
null_files = []
invalid_files = []
edited_counter = 0
edit_record = []

for i in range(0, height - 60, 400):
    for j in range(0, width - 60, 300):
        edited_counter = 0
        counter_for_backgrounds += 2

        for file in flags:
            image = load_image_with_retry(file)
            if image is None:
                null_files.append(file)
                continue
            try:
                flag_h, flag_w = image.shape[:2]
            except AttributeError:
                invalid_files.append(file)
                continue

            # # ======= تأثير بكسلة (تم التعليق عليه) =======
            # scale = random.choice([1])
            # small = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv2.INTER_NEAREST)
            # pixelated = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # ======= Resize نهائي =======
            final_w = random.choice(FINAL_SHAPE)
            final_h = random.choice(FINAL_SHAPE)
            resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_LINEAR)

            # # ======= Gaussian Blur (مُعطل) =======
            kernal = random.choice(gaussian_list)
            blurred = cv2.GaussianBlur(resized, (kernal, kernal), 0)
            final = blurred

            # ======= Noise (مُعطل) =======
            noise_param = random.choice(noises)
            noise = np.random.normal(0, noise_param, resized.shape).astype(np.uint8)
            final = cv2.add(final, noise)

            # # ======= نسخة نظيفة بدون تأثيرات ضارة =======
            # final = resized

            # تدوير/قلب عشوائي
            final = random_flip_or_rotate(final)

            # تعديل السطوع
            brightness = random.choice(BRIGHTNESS_FACTORS)
            darker = np.clip(final * brightness, 0, 255).astype(np.uint8)


            background = random.choice(backgrounds).copy()
            counter_for_backgrounds += 1

            if i + darker.shape[0] <= height and j + darker.shape[1] <= width:
                roi = background[i:i + darker.shape[0], j:j + darker.shape[1]]
                if roi.shape[:2] == darker.shape[:2]:
                    alpha = random.uniform(0.85, 0.95)
                    blended = cv2.addWeighted(darker, alpha, roi, 1 - alpha, 0)
                    background[i:i + darker.shape[0], j:j + darker.shape[1]] = blended
                    # rotate img
                    (h,w) = background.shape[:2]
                    seta = random.choice(angle)
                    rotate_matrix = cv2.getRotationMatrix2D((w//2 , h//2),seta ,1)
                    rotated = cv2.warpAffine(background , rotate_matrix ,(w , h) )
                    bluer = random.choice(gaussian_list)
                    rotated = cv2.GaussianBlur(rotated, (kernal, kernal), 0)
                    end = rotated

                    filename = os.path.splitext(os.path.basename(file))[0]
                    output_path = os.path.join(EDITED_FLAGS, f"{filename}_{i}_{j}.jpg")
                    cv2.imwrite(output_path, end)
                    print(f"✅ Saved: {output_path}")
                    edited_counter += 1

        edit_record.append(edited_counter)

# ===== حفظ صور خلفية بدون أعلام =====
for idx in range(NUM_BACKGROUNDS_TO_SAVE):
    bg = backgrounds[idx % len(backgrounds)].copy()
    output_path = os.path.join(EDITED_FLAGS, f"background_only_{idx}.jpg")
    cv2.imwrite(output_path, bg)
    if idx % 50 == 0:
        print(f"📷 Background only image saved: {output_path}")

# ===== التقارير =====
print("-" * 50)
print(f"📦 عدد الملفات اللي فشلت: {len(null_files)}")
if null_files:
    print("⚠️ Ignored files:")
    for f in null_files:
        print(f"- {f}")
print(f"🖼️ عدد الصور المعدلة في كل مجموعة: {np.unique(np.array(edit_record))}")
