import os
import random
import shutil

path_image = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\images"
path_images_train = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\images\train"
path_images_val = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\images\val"

os.makedirs(path_image , exist_ok=True)

path_annotation = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\labels"
path_annotation_train = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\labels\train"
path_annotation_val = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\flage_on desert_not_roat\labels\val"

os.makedirs(path_annotation , exist_ok=True)


create_folders = [
    path_images_train,
    path_images_val,
    path_annotation_train,
    path_annotation_val,
]


all_images = [
    f for f in os.listdir(path_image) if os.path.isfile(os.path.join(path_image, f))
]
all_annotation = [
    f
    for f in os.listdir(path_annotation)
    if os.path.isfile(os.path.join(path_annotation, f))
]

random.seed(45)
random.shuffle(all_images)
random.shuffle(all_annotation)

for folder in create_folders:
    os.makedirs(folder, exist_ok=True)

num_train_part = int(len(all_images) * 0.8)

for idx, image in enumerate(all_images):
    if idx < num_train_part:
        shutil.move(
            os.path.join(path_image, image), os.path.join(path_images_train, image)
        )
        # Move annotation with the same base name
        ann_name = os.path.splitext(image)[0] + ".txt"
        ann_src = os.path.join(path_annotation, ann_name)
        ann_dst = os.path.join(path_annotation_train, ann_name)
        if os.path.exists(ann_src):
            shutil.move(ann_src, ann_dst)
    else:
        shutil.move(
            os.path.join(path_image, image), os.path.join(path_images_val, image)
        )
        # Move annotation with the same base name
        ann_name = os.path.splitext(image)[0] + ".txt"
        ann_src = os.path.join(path_annotation, ann_name)
        ann_dst = os.path.join(path_annotation_val, ann_name)
        if os.path.exists(ann_src):
            shutil.move(ann_src, ann_dst)
print("✅ Script finished successfully.")
