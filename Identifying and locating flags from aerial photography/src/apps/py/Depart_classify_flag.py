import os , shutil




flag_classification_path = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\flag_classification"


flags = os.listdir(flag_classification_path)

for flag in flags:
    flag_folder = os.path.join(flag_classification_path , flag.split('_')[0])
    os.makedirs(flag_folder ,exist_ok=True)
    flag_path = os.path.join(flag_classification_path , flag)
    shutil.move(flag_path , flag_folder)


# dirs = os.listdir(flag_classification_path)

# for dir in dirs :
#     folder = os.path.join(flag_classification_path , dir)
#     if os.path.isdir(folder):
#         flags = os.listdir(folder)
#         for flag in flags :
#             src = os.path.join(flag_classification_path , dir , flag) 
#             dst = os.path.join(flag_classification_path , flag)
#             shutil.move(src , dst)