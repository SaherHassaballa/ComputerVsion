import os 
import shutil

file_needed_img_path = r"C:\Users\saher\Desktop\saher.txt"
all_img_path = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\inserted_flags_in_background\images"
distination = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\yes_no\images" 

with open(file_needed_img_path , 'r') as f :
    lines = f.readlines()
    for file in lines :
        for i , img in enumerate(os.listdir(all_img_path)):
            if img == file.strip() :
                src = os.path.join(all_img_path , img)
                dst = os.path.join(distination , img)
                shutil.copy(src , dst)
                print(f"{i+1} done")