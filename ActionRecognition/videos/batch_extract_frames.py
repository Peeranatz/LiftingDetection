import os
import cv2

def build_sequences(input_root="/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/videos/extracted_frames", output_root="final_dataset", seq_len=15, step=5):
    for action in os.listdir(input_root):
        action_dir = os.path.join(input_root, action)
        if not os.path.isdir(action_dir):
            continue

        for clip in os.listdir(action_dir):
            clip_dir = os.path.join(action_dir, clip)
            if not os.path.isdir(clip_dir):
                continue

            images = sorted([
                f for f in os.listdir(clip_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            total_frames = len(images)
            seq_count = 0
            
            print(f"Found {total_frames} frames in {clip_dir}")

            for i in range(0, total_frames - seq_len + 1, step):
                seq_images = images[i:i+seq_len]
                seq_folder = os.path.join(output_root, action, clip, f"seq{seq_count:04d}")
                os.makedirs(seq_folder, exist_ok=True)

                for j, img_name in enumerate(seq_images):
                    src_path = os.path.join(clip_dir, img_name)
                    dst_path = os.path.join(seq_folder, f"{j:03d}.jpg")
                    img = cv2.imread(src_path)
                    cv2.imwrite(dst_path, img)

                seq_count += 1

            print(f"✅ {action}/{clip} → {seq_count} sequences")

if __name__ == "__main__":
    build_sequences()
