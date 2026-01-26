import os

folder_path = r"D:\github lite\cv_projects\Identifying and locating flags from aerial photography\data\downloaded_flags"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def sanitize_filename(filename):
    """Replace special characters with underscores and ensure uniqueness."""
    sanitized = (
        filename.replace(" ", "_")
        .replace("'", "_")
        .replace("é", "e")
        .replace("ç", "c")
        .replace("å", "a")
        .replace("ö", "o")
        .replace("ô", "o")
        .replace("ü", "u")
    )
    return sanitized


def rename_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        old_path = os.path.join(folder_path, file)
        if os.path.isfile(old_path):
            new_name = sanitize_filename(file)
            new_path = os.path.join(folder_path, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


rename_files_in_folder(folder_path)
