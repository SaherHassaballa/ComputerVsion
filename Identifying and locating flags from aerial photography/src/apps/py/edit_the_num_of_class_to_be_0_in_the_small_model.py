import os

labels_dir = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\detect_flag_or_not\labels"

for split in ["train", "val"]:
    split_path = os.path.join(labels_dir, split)
    for filename in os.listdir(split_path):
        if filename.endswith(".txt"):
            path = os.path.join(split_path, filename)
            with open(path, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"  # تحويل الكلاس لـ 0
                    new_lines.append(" ".join(parts))
            with open(path, "w") as f:
                f.write("\n".join(new_lines))
