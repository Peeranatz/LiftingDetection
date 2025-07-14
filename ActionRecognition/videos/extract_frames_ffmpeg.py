import subprocess
import os

def extract_frames(video_path, output_dir="extracted_frames", prefix="frame"):
    if not os.path.isfile(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # สร้างชื่อไฟล์รูปภาพ เช่น: frame_001.jpg, frame_002.jpg
    output_pattern = os.path.join(output_dir, f"{prefix}_%03d.jpg")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        output_pattern
    ]

    try:
        print(f"🚀 Extracting frames from: {video_path}")
        subprocess.run(cmd, check=True)
        print(f"✅ Done! Images saved to: {output_dir}")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Frame extraction failed.")

if __name__ == "__main__":
    # กำหนด path ของไฟล์วิดีโอที่ต้องการตัด
    input_video = "/Users/balast/Desktop/LiftingProject/LiftingDetection/video_datasets/Carrying/Datatest1.mp4"
    
    # จะได้รูปภาพในโฟลเดอร์นี้
    extract_frames(input_video, output_dir="extracted_frames", prefix="carry_normal")
