import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import datetime as dt
import matplotlib.pyplot as plt

from datetime import datetime
from ultralytics import YOLO
from collections import deque, namedtuple, OrderedDict
from keras.models import load_model
from Database_system.models.action_model import Action

yolo_human = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/human.pt"
)
yolo_box = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/box.pt"
)

SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.5
SMOOTH_FRAMES = 5  # number of frames for landmark smoothing
MIN_VALID_FRAMES = 15

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

id_logs = {}

global_id_counter = 1

# human_profile = {}     # track_id -> { last_seen, action, box_id }
last_action = {}       # track_id -> last action label
action_start = {}      # track_id -> datetime of start
buffers = {}           # track_id -> deque
pose_instances = {}    # track_id -> Pose object
landmark_history = {}  # track_id -> deque for smoothing landmarks

known_tracks = {}   # track_id (from yolo) -> global_id
last_seen = {}      # track_id -> timestamp
timeout_frames = 30  # number of frames to forget track_id

# target_id = 1
# debug_buffer = list()

SELECTED_JOINTS = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,
    25,
    26,  # left knee, right knee
    27,
    28,  # left ankle, right ankle
    29,
    30,  # left heel, right heel
    31,
    32,  # left foot index, right foot index
]

def collect_pose_landmarks(buffer: deque, landmarks):
    pose_array = []
    for j in SELECTED_JOINTS:
        p = landmarks[j]
        pose_array.append([p.x, p.y])

    buffer.append(np.array(pose_array))


def get_action(buffer: deque, std_threshold: float = 0.015) -> str:
    if len(buffer) < 10:
        return "unknown"

    arr = np.array(buffer)
    avg_std = np.mean(np.std(arr, axis=0))
    
    return ("idle" if avg_std < std_threshold else "moving"), avg_std

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

def point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox 
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_bbox(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.1):
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(img_w, x2 + pad_w)
    ny2 = min(img_h, y2 + pad_h)
    return nx1, ny1, nx2, ny2


def plot_joint_std(buffer: deque):
    arr = np.array(buffer)  # (N_frames, N_joints, 2)
    joint_std = []

    for j in range(arr.shape[1]):
        joint_data = arr[:, j, :]  # (N_frames, 2)
        std_xy = np.std(joint_data, axis=0)
        joint_std.append(std_xy)

    joint_std = np.array(joint_std)

    joints = list(range(arr.shape[1]))
    plt.figure(figsize=(10, 5))
    plt.plot(joints, joint_std[:, 0], label="std_x", marker="o")
    plt.plot(joints, joint_std[:, 1], label="std_y", marker="x")
    plt.title("Std Dev per Joint (x and y)")
    plt.xlabel("Joint Index (25-32 → 0-7)")
    plt.ylabel("Std Dev")
    plt.legend()
    plt.grid(True)
    plt.show()


def log_action(person_id, action, start_time, end_time, object_type=None):
    act = Action(
        person_id=person_id,
        action=action,
        object_type=object_type,
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    act.save()
    print("✅ Logged Action:", person_id, action, object_type)

cap = cv.VideoCapture(1)
# cap = cv.VideoCapture(
#     "/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/data/test_video/test_video_4.mp4"
# )
pTime = 0
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame = cv.flip(frame, 1)
    if not success:
        continue
    
    frame_idx += 1

    current_time = time.time()
    # Detect humans
    human_res = yolo_human.track(source=frame, stream=False, tracker="/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/tracker/bytetrack.yaml")[0]
    human_bboxes = {}
    for person in human_res.boxes:
        hconf = person.conf[0].item()
        if hconf < CONF_THRESHOLD:
            continue
        
        track_id = int(person.id[0]) if person.id is not None else -1
        hx1, hy1, hx2, hy2 = map(int, person.xyxy[0])
        # x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
        area = (hx2 - hx1) * (hy2 - hy1)
        
        if track_id in known_tracks:
            global_id = known_tracks[track_id]
        else:
            global_id = global_id_counter
            known_tracks[track_id] = global_id
            global_id_counter += 1
            
        last_seen[track_id] = frame_idx
        
        # เก็บข้อมูลเบื้องต้นของแต่ละ global_id
        id_logs.setdefault(global_id, {
            "actions": [],
            "last_seen_frame": frame_idx,
            "total_frames": 0,
        })

        # อัปเดตข้อมูลทุกเฟรมที่เจอ
        id_logs[global_id]["last_seen_frame"] = frame_idx
        id_logs[global_id]["total_frames"] += 1
        
        if id_logs[global_id]["total_frames"] < MIN_VALID_FRAMES:
            continue
                        
        # เก็บเฉพาะ box ที่ใหญ่สุดของ track_id นี้
        if (global_id not in human_bboxes) or (area > human_bboxes[global_id]['area']):
            human_bboxes[global_id] = {
                "bbox": (hx1, hy1, hx2, hy2),
                "conf": hconf,
                "area": area
            }
        cv.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        
        # เตรียม buffer ถ้ายังไม่มี
        buffers.setdefault(global_id, deque(maxlen=SEQUENCE_LENGTH))
        last_action.setdefault(global_id, None)
        action_start.setdefault(global_id, None)
        
        if global_id not in pose_instances:
            pose_instances[global_id] = mpPose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        
        landmark_history.setdefault(global_id, deque(maxlen=SMOOTH_FRAMES))
        
        # Pose ROI
        roi = frame[hy1:hy2, hx1:hx2]
        if roi.size == 0: continue
        roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
        pose_results = pose_instances[global_id].process(roi_rgb)
        if not pose_results.pose_landmarks: continue
        lms = pose_results.pose_landmarks.landmark
        collect_pose_landmarks(buffers[global_id], lms)
        
        # collect raw full landmarks for smoothing
        raw_pts = np.array([(lm.x, lm.y) for lm in lms])
        landmark_history[global_id].append(raw_pts)
        
        # smooth landmarks if available
        if len(landmark_history[global_id]) > 0:
            hist = np.stack(landmark_history[global_id], axis=0)
            smooth_pts = np.mean(hist, axis=0)  # (33,2)
        else:
            smooth_pts = raw_pts
        
        # draw smoothed skeleton on ROI
        for (start, end) in mpPose.POSE_CONNECTIONS:
            x1n, y1n = smooth_pts[start]
            x2n, y2n = smooth_pts[end]
            p1 = mpDraw._normalized_to_pixel_coordinates(x1n, y1n, hx2-hx1, hy2-hy1)
            p2 = mpDraw._normalized_to_pixel_coordinates(x2n, y2n, hx2-hx1, hy2-hy1)
            if p1 and p2:
                cv.line(roi, p1, p2, (0, 255, 255), 2)
        for nx, ny in smooth_pts:
            px, py = int(nx*(hx2-hx1)), int(ny*(hy2-hy1))
            cv.circle(roi, (px, py), 3, (255, 255, 0), -1)
            
        action_label = "unknown"
        avg = 0
        
        if len(buffers[global_id]) == SEQUENCE_LENGTH:
            action_label, avg = get_action(buffers[global_id])
            now = dt.datetime.now()
            
            box_res = yolo_box.track(source=frame, stream=False, tracker="/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/tracker/bytetrack.yaml")[0]
            for box in box_res.boxes:
                bconf = float(box.conf[0])
                if bconf < CONF_THRESHOLD:
                    continue
                
                cls = int(box.cls[0])
                object_label = box_res.names[cls]
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                
                # เตรียมจุดของกล่อง
                points = [
                    ((bx1 + bx2) // 2, (by1 + by2) // 2),  # center
                    (bx1, by1),  # top-left
                    (bx2, by1),  # top-right
                    (bx1, by2),  # bottom-left
                    (bx2, by2),  # bottom-right
                    ((bx1 + bx2) // 2, by1),  # top-center
                    ((bx1 + bx2) // 2, by2),  # bottom-center
                    (bx1, (by1 + by2) // 2),  # middle-left
                    (bx2, (by1 + by2) // 2),  # middle-right
                ]
                
                matched = any(point_in_bbox(px, py, (hx1, hy1, hx2, hy2)) for (px, py) in points)
                iou_val = iou([bx1, by1, bx2, by2], [hx1, hy1, hx2, hy2])
                # print("IOU value (Box vs Human {}): {:.3f}".format(global_id, iou_val))
                # print("Matched : {}".format(matched))
                
                cv.rectangle(frame, (bx1,by1), (bx2,by2), (255,0,0), 2)
                cv.putText(frame,
                        f"{object_label} {bconf:.2f}",
                        (bx1, by1-5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                
                if matched or iou_val > 0.1:
                    action_label = "carrying"
                    break  # ไม่ต้องเช็คกล่องอื่นแล้ว
            
            id_logs[global_id]["actions"].append(action_label)
            
            if action_label == 'carrying':
                # print("ID: {} | Action: {} | Object type: {}".format(global_id, action_label, object_label))
                cv.putText(
                frame,
                f"ID:{global_id} | {hconf:.2f} | Action: {action_label} | Object: {object_label}",
                (hx1, hy1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3,
            )
            else:    
                # print("ID: {} | Action: {}".format(global_id, action_label))
                cv.putText(
                    frame,
                    f"ID:{global_id} | {hconf:.2f} | Action: {action_label} | Avg: {avg:.2f}",
                    (hx1, hy1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3,
                )
        else:
            cv.putText(
                    frame,
                    f"ID:{global_id} | {hconf:.2f}",
                    (hx1, hy1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3,
                )
            
    
    to_delete = []
    for tid, t in last_seen.items():
        if frame_idx - t > timeout_frames:
            to_delete.append(tid)

    for tid in to_delete:
        gid = known_tracks.get(tid)
        known_tracks.pop(tid, None)
        last_seen.pop(tid, None)
        if gid is not None:
            buffers.pop(gid, None)
            pose_instances.pop(gid, None)
            landmark_history.pop(gid, None)
            last_action.pop(gid, None)
            action_start.pop(gid, None)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(
        frame, f"FPS: {int(fps)} | FRAME: {int(frame_idx)}", (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
    )

    cv.imshow("Multi-Person pose", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

print("Final Action Logs:")
for gid, info in id_logs.items():
    if info["total_frames"] < MIN_VALID_FRAMES:
        continue
    
    print(f"ID {gid} | Last seen frame: {info['last_seen_frame']} | Total frames: {info['total_frames']}")
    if info["actions"]:
        print(f"Actions: {info['actions']}")
    else:
        print("Actions: [No actions recorded]")
    print("-" * 50)