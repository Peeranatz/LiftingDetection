import cv2 as cv
import pandas as pd
import numpy as np
import mediapipe as mp
import time

from ultralytics import YOLO
from collections import deque, namedtuple, OrderedDict 

yolo_model = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/best_v2.pt"
)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture(0)

pTime = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    yolo_results = yolo_model.track(source=frame, stream=False, tracker="bytetrack.yaml")[0]
    
    for box in yolo_results.boxes:
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())  # ดึง index ของคลาสที่ตรวจพบเจอ
        label = yolo_model.names[cls]

        if conf < 0.5 or label != 'human':
            continue
        
        track_id = int(box.id[0]) if box.id is not None else -1 
        print(f"ID: {track_id} | Label: {label}")

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.putText(
            frame,
            f"ID:{track_id} | {label} {conf:.2f}",
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        if label == 'human':
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
            pose_results = pose.process(roi_rgb)
            if not pose_results.pose_landmarks:
                continue

            h_roi, w_roi, _ = roi.shape
            for lm in pose_results.pose_landmarks.landmark:
                # พิกัดใน ROI
                cx_roi = int(lm.x * w_roi)
                cy_roi = int(lm.y * h_roi)
                # offset กลับไปยังภาพหลัก
                cx = x1 + cx_roi
                cy = y1 + cy_roi
                cv.circle(frame, (cx, cy), 4, (0, 0, 255), cv.FILLED)
            
            mpDraw.draw_landmarks(roi, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Multi-Person pose", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()