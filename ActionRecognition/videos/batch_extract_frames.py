import os
import subprocess

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%03d.jpg")

    cmd = ["ffmpeg", "-i", video_path, output_pattern]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Extracted: {video_path} → {output_dir}")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed: {video_path}")

def process_all_videos(input_root="/Users/balast/Desktop/LiftingProject/LiftingDetection/video_datasets_2", output_root="extracted_frames"):
    for action in os.listdir(input_root):
        action_dir = os.path.join(input_root, action)
        if not os.path.isdir(action_dir):
            continue

        for fname in os.listdir(action_dir):
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(action_dir, fname)
            clip_name = os.path.splitext(fname)[0]
            output_dir = os.path.join(output_root, action, clip_name)

            extract_frames(video_path, output_dir)

if __name__ == "__main__":
    process_all_videos()

