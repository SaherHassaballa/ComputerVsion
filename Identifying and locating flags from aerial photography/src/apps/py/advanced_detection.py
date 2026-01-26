# import flagpy as fp

# # لو عندك مسار ملف، هتحتاج ترفعه كـ path URL
# path = r"C:\Users\saher\Pictures\Screenshots\Screenshot 2025-07-10 220616.png"
# url = "file://" + path.replace("\\", "/")
# country = fp.identify(url, method="mse")  # أو ssim، hash
# print(country)

# import nyckel

# # تسجيل الدخول
# credentials = nyckel.Credentials("8k0j007a0554nnfxgut341z4aogkpb6o", "hyhdj046by71l0mkc1wz9or7lw8bjmlpzgnp62dgqms5uyia2w8oe6es7asnrsmo")
# flag_api = "aboriginal-flags-identifier"  # حسب وثائق Nyckel

# # تحليل الرابط أو الصورة
img = r"C:\Users\saher\Pictures\Screenshots\Screenshot 2025-07-10 220616.png"
# result = nyckel.invoke(flag_api, img, credentials)
# print(result)

from flagsense.yolo_inference import Detector
import os

model = Detector(model_name="v10")
out = r"C:\flag_results"
os.makedirs(out, exist_ok=True)

results = model.detect(img, save_dir=out)
for res in results:
    for d in res.detections:
        print(d["label"], d["confidence"])
