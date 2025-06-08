import cv2 
import mediapipe as mp
import time
import numpy as np
import datetime as dt
import csv
from keras.models import load_model
from collections import deque, namedtuple

# ===== Configuration =====
MODEL_PATH = "/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/models/lstm_model.h5"
SHOULDER_WIDTH_M = 0.312
SEQUENCE_LENGTH = 30
ACTION_LABELS = ["standing", "moving", "carrying"]  # ปรับให้ตรงกับที่เทรน
OUTPUT_CSV = "action_log.csv"
DEBUG = False  # True เพื่อ debug prints

# ===== Mediapipe Setup =====
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
Point = namedtuple("Point", ["x", "y", "z"])

# ===== Selected joints (NTU schema) =====
SELECTED_JOINTS = [
    14, 16, 20, 22, 18,      # Right arm
    13, 15, 19, 17, 21,      # Left arm
    24, 26, 28, 23, 25, 27,  # Legs
    "center_shoulder", 0, "center_hip"
]

# ===== Helper functions =====

def calculate_m_per_pixel(l_sh: Point, r_sh: Point, img_w: int, img_h: int) -> float:
    """Return meters per pixel by comparing shoulder width."""
    dx = (l_sh.x - r_sh.x) * img_w
    dy = (l_sh.y - r_sh.y) * img_h
    dist = np.hypot(dx, dy)
    return (SHOULDER_WIDTH_M / dist) if dist else 1.0

def get_midpoint(a: Point, b: Point) -> Point:
    """Return midpoint of two landmarks."""
    return Point((a.x + b.x) / 2,
                 (a.y + b.y) / 2,
                 (a.z + b.z) / 2)

def normalize_to_ntu(landmarks, img_w: int, img_h: int):
    """
    Normalize mediapipe landmarks to NTU coordinate system:
    - center at hip midpoint
    - invert y-axis
    - scale to real-world meters
    """
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mpp = calculate_m_per_pixel(l_sh, r_sh, img_w, img_h)

    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_c = get_midpoint(l_hip, r_hip)
    sh_c = get_midpoint(l_sh, r_sh)

    joints_list = []
    for j in SELECTED_JOINTS:
        p = None
        if isinstance(j, int):
            p = landmarks[j]
        elif j == "center_shoulder":
            p = sh_c
        else:  # center_hip
            p = hip_c

        x_px = p.x * img_w
        y_px = p.y * img_h
        z_px = p.z * img_w

        cx = hip_c.x * img_w
        cy = hip_c.y * img_h

        x_m = (x_px - cx) * mpp
        y_m = -(y_px - cy) * mpp
        z_m = z_px * mpp

        joints_list.append([x_m, y_m, z_m])

    return np.array(joints_list), mpp

def predict_action(buffer: deque, model) -> str:
    """Predict action label from a sequence buffer."""
    seq = np.array(buffer).reshape(1, SEQUENCE_LENGTH, -1)
    pred = model.predict(seq, verbose=0)
    return ACTION_LABELS[int(np.argmax(pred))]

def save_action_log(logs: list, path: str) -> None:
    """Save action log to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["action", "start_time", "end_time"])
        writer.writerows(logs)

# ===== Main Application =====
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❗ Cannot open camera")
        return

    model = load_model(MODEL_PATH, compile=False)
    buffer = deque(maxlen=SEQUENCE_LENGTH)
    last_action = None
    action_start = None
    logs = []
    p_time = time.time()

    with mp_pose.Pose() as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                h, w, _ = frame.shape
                lm = results.pose_landmarks.landmark
                ntu_pose, mpp = normalize_to_ntu(lm, w, h)
                buffer.append(ntu_pose.flatten())

                if len(buffer) == SEQUENCE_LENGTH:
                    label = predict_action(buffer, model)
                    now = dt.datetime.now()

                    if label != last_action:
                        if last_action:
                            logs.append([last_action, action_start, now])
                            if DEBUG:
                                print(f"Logged: {last_action} | {action_start} → {now}")
                        last_action = label
                        action_start = now

                    cv2.putText(frame, f"Action: {label}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Display FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time) if p_time else 0
            p_time = c_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("ActionRecognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Finalize log
    if last_action and action_start:
        end = dt.datetime.now().isoformat()
        logs.append([last_action, action_start, end])
        if DEBUG:
            print(f"Final log: {last_action} | {action_start} → {end}")

    print("\nAction Summary:")
    for act, start, end in logs:
        print(f"{act:10s} | {start} → {end}")

if __name__ == '__main__':
    main()

