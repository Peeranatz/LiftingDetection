import cv2
import json
import yaml
import numpy as np
import os

from ultralytics import YOLO
from collections import defaultdict
from typing import Dict, Any

# step 1
def collect_raw_keypoints(config_path: str, output_json: str = "raw_keypoints.json") -> Dict[int, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["HUMAN_MODEL_PT"]
    video_source = config["VIDEO_SOURCE"]


    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_source}")

    pose_sequences = defaultdict(list)
    frame_id = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True)

        if len(results) == 0 or results[0].boxes.id is None:
            frame_id += 1
            continue

        for keypoints, track_id in zip(results[0].keypoints.data.cpu().numpy(), results[0].boxes.id):
            if track_id is None:
                continue

            track_id = int(track_id.item())
            keypoints_flat = keypoints.flatten().tolist()  # shape (17*3,)
            pose_sequences[track_id].append(keypoints_flat)

        frame_id += 1
        
        cv2.imshow("YOLOv11 Pose Detection", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit by user.")
            break 

    cap.release()

    # Build output dictionary
    output_data = {}
    for pid, seq in pose_sequences.items():
        # จาก seq แต่ละเฟรม = list ยาว 51 ค่า
        reshaped_seq = [np.array(frame).reshape(17, 3).tolist() for frame in seq]

        output_data[pid] = {
            "frames": len(reshaped_seq),
            "keypoints": reshaped_seq
        }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved {len(output_data)} person sequences → {output_json}")
    return output_data


# step 2
def normalize_sequence_length(input_json: str, output_dir: str = "aligned_npy",target_length: int = 60) -> Dict[int, Any]:
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r") as f:
        data = json.load(f)
        
    summary = {}
    
    def sample_or_pad(seq: np.ndarray, target_len: int) -> np.ndarray:
        n = len(seq)
        if n == target_len:
            return seq
        elif n > target_len:
            # Uniform sampling
            idx = np.linspace(0, n - 1, target_len).astype(int)
            return seq[idx]
        else:
            # Pad with zeros (T,17,3)
            pad = np.zeros((target_len - n, seq.shape[1], seq.shape[2]))
            return np.concatenate([seq, pad], axis=0)
    
    for pid, info in data.items():
        seq = np.array(info["keypoints"])  # (T,17,3)
        T_original = seq.shape[0]
        seq_aligned = sample_or_pad(seq, target_length)

        np.save(f"{output_dir}/track_{pid}.npy", seq_aligned)
        summary[pid] = {
            "original_frames": T_original,
            "aligned_frames": target_length,
            "file": f"track_{pid}.npy"
        }

        print(f"Track {pid}: {T_original} → {target_length} frames")
        
        summary_path = os.path.join(output_dir, "alignment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"Alignment complete! Saved {len(summary)} sequences in '{output_dir}/'")
        return summary

if __name__ == "__main__":
    data = collect_raw_keypoints(
        config_path="/Users/balast/Desktop/Desktop - All file/LiftingProject/LiftingDetection/ActionRecognition/GNN/config.yaml",
        output_json="raw_keypoints.json"
    )
    
    normalize_sequence_length(
        input_json="raw_keypoints.json",
        output_dir="aligned_npy",
        target_length=60
    )
