import cv2 as cv
import pandas as pd
import numpy as np
import mediapipe as mp
import time
import datetime as dt

from ultralytics import YOLO
from collections import deque, namedtuple, OrderedDict 
from keras.models import load_model

yolo_model = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/best_v2.pt"
)
action_model = load_model("/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/models/lstm_model.h5")

SHOULDER_WIDTH_M = 0.312
SEQUENCE_LENGTH = 30
ACTION_LABELS = ["standing", "moving", "carrying"]  # ปรับให้ตรงกับที่เทรน
CONF_THRESHOLD   = 0.5
DEBUG = False

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
Point = namedtuple("Point", ["x", "y", "z"])

buffers = {}

SELECTED_JOINTS = [
    14, 16, 20, 22, 18,      # Right arm
    13, 15, 19, 17, 21,      # Left arm
    24, 26, 28, 23, 25, 27,  # Legs
    "center_shoulder", 0, "center_hip"
]

cap = cv.VideoCapture(1)
pTime = 0

def calculate_m_per_pixel(l_sh: Point, r_sh: Point, img_w: int, img_h: int) -> float:
    dx = (l_sh.x - r_sh.x) * img_w
    dy = (l_sh.y - r_sh.y) * img_h
    dist = np.hypot(dx, dy)
    return (SHOULDER_WIDTH_M / dist) if dist else 1.0

def get_midpoint(a: Point, b: Point) -> Point:
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
    l_sh = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value]
    mpp = calculate_m_per_pixel(l_sh, r_sh, img_w, img_h)

    l_hip = landmarks[mpPose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mpPose.PoseLandmark.RIGHT_HIP.value]
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
    seq = np.array(buffer).reshape(1, SEQUENCE_LENGTH, -1)
    pred = model.predict(seq, verbose=0)
    return ACTION_LABELS[int(np.argmax(pred))]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    yolo_results = yolo_model.track(source=frame, stream=False, tracker="bytetrack.yaml")[0]
    
    for box in yolo_results.boxes:
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())  # ดึง index ของคลาสที่ตรวจพบเจอ
        label = yolo_model.names[cls]

        if conf < CONF_THRESHOLD:
            continue
        
        track_id = int(box.id[0]) if box.id is not None else -1 
        # print(f"ID: {track_id} | Label: {label}")

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if label == 'human':
            if track_id not in buffers:
                buffers[track_id] = deque(maxlen=SEQUENCE_LENGTH)
            
            last_action = None 
            action_start = None
            logs = []
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
            pose_results = pose.process(roi_rgb)
            if not pose_results.pose_landmarks:
                continue
            
            mpDraw.draw_landmarks(roi, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            h_roi, w_roi, _ = roi.shape
            lm = pose_results.pose_landmarks.landmark
            ntu_pose, mpp = normalize_to_ntu(lm, w_roi, h_roi)
            buffers[track_id].append(ntu_pose)
            
            if len(buffers[track_id]) == SEQUENCE_LENGTH:
                action_label = predict_action(buffers[track_id], action_model)
                print("--------------------------------------------------")
                now = dt.datetime.now()
                
                if action_label != last_action:
                    if last_action:
                        logs.append([last_action, action_start, now])
                        if DEBUG:
                            print(f"Logged: {last_action} | {action_start} → {now}")
                    last_action = action_label 
                    action_start = now
                
                cv.putText(
                frame,
                f"ID:{track_id} | {label} {conf:.2f} | Action: {action_label}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        
        else:
            cv.putText(
                frame,
                f"ID:{track_id} | {label} {conf:.2f}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, f"FPS: {int(fps)}", (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Multi-Person pose", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

  # for lm in pose_results.pose_landmarks.landmark:
                # พิกัดใน ROI
                # cx_roi = int(lm.x * w_roi)
                # cy_roi = int(lm.y * h_roi)
                # offset กลับไปยังภาพหลัก
                # cx = x1 + cx_roi
                # cy = y1 + cy_roi
                # cv.circle(frame, (cx, cy), 4, (0, 0, 255), cv.FILLED)