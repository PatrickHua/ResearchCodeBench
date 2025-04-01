import os
import shutil
import filecmp
def ignore_git(src, names):
    return [name for name in names if name == '.git']

def sync_folders(src, dst):
    """Sync files from src to dst, copying only changed or missing files."""
    
    # If destination doesn't exist, copy everything and return
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(f"Copied entire folder: {src} -> {dst}")
        return

    # Compare the directories
    comparison = filecmp.dircmp(src, dst)

    # Files only in source -> Copy them
    for file in comparison.left_only + comparison.diff_files:
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
            print(f"Copied: {src_path} -> {dst_path}")

    # Recursively sync subdirectories
    for subdir in comparison.common_dirs:
        sync_folders(os.path.join(src, subdir), os.path.join(dst, subdir))

# Example usage
# sync_folders("source_folder", "destination_folder")