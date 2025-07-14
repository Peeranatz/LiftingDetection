import os
import cv2

def build_sequences(image_dir, output_dir, label, seq_len=15, step=5):
    os.makedirs(output_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    total_frames = len(images)
    seq_count = 0

    for i in range(0, total_frames - seq_len + 1, step):
        seq_images = images[i:i+seq_len]
        seq_folder = os.path.join(label_dir, f"seq{seq_count:04d}")
        os.makedirs(seq_folder, exist_ok=True)

        for j, img_name in enumerate(seq_images):
            src_path = os.path.join(image_dir, img_name)
            dst_path = os.path.join(seq_folder, f"{j:03d}.jpg")
            img = cv2.imread(src_path)
            cv2.imwrite(dst_path, img)

        seq_count += 1

    print(f"✅ Done: Created {seq_count} sequences in '{label_dir}'")

if __name__ == "__main__":
    input_folder = "/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/videos/extracted_frames"
    output_folder = "final_dataset"
    label = "carry_normal"

    build_sequences(input_folder, output_folder, label)
