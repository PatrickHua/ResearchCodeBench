import os
import shutil
import glob
import datetime

def get_timestamped_output_dir(base_output_dir):
    # Create a timestamp directory name
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Create the base output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Look for most recent timestamped directory
    timestamp_dirs = glob.glob(os.path.join(base_output_dir, "*-*-*-*-*-*"))
    
    # Create the new timestamped directory
    new_output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(new_output_dir, exist_ok=True)
    
    # If there are previous timestamped directories, copy the most recent one
    if timestamp_dirs:
        # Sort by creation time (most recent last)
        timestamp_dirs.sort(key=os.path.getctime)
        most_recent_dir = timestamp_dirs[-1]
        
        # Only copy if the directory exists and is not the same as our new one
        if os.path.exists(most_recent_dir) and most_recent_dir != new_output_dir:
            print(f"Copying contents from most recent run: {os.path.basename(most_recent_dir)}")
            for item in os.listdir(most_recent_dir):
                source = os.path.join(most_recent_dir, item)
                destination = os.path.join(new_output_dir, item)
                if os.path.isdir(source):
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
    
    return new_output_dir