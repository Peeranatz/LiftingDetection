import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import datetime as dt
import matplotlib.pyplot as plt

from datetime import datetime
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from collections import deque
from Database_system.models.action_model import Action

yolo_human = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/human.pt"
)
yolo_box = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/box.pt"
)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

# Constants
SEQUENCE_LENGTH   = 50  # จำนวน frame ในการตัดสินใจ action 
MIN_VALID_FRAMES  = 15  # ถ้าคนปรากฎน้อยกว่า 15 เฟรมจะไม่เชื่อถือ
CONF_THRESHOLD    = 0.5 
SMOOTH_FRAMES     = 5   # smoothing pose

MAX_DIST          = 100    # px สำหรับ matching centroid
DROPOUT_THRESHOLD = 50    # ถ้าไม่เจอคนนานเกินนี้ -> ลบทิ้ง 

# State
id_logs    = {}              # gid -> {"actions":[], "last_seen_frame":…, "total_frames":…}
buffers    = {}              # gid -> deque for pose seq
pose_instances   = {}        # gid -> mpPose.Pose()
landmark_history = {}        # gid -> deque for smoothing
tracks     = {}              # gid -> {"last_centroid":(x,y), "missed":int}
assign_map = {}              # gid -> (x1,y1,x2,y2)
next_gid   = 1

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

# cap = cv.VideoCapture(1)
cap = cv.VideoCapture(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/data/test_video/test_video_4.mp4"
)
pTime = 0
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    # frame = cv.flip(frame, 1)
    frame_idx += 1
    
    # Detect humans
    human_res = yolo_human.track(source=frame, stream=False, tracker="/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/tracker/bytetrack.yaml")[0]
    detections = []
    for person in human_res.boxes:
        hconf = float(person.conf[0])
        if hconf < CONF_THRESHOLD: continue
        x1,y1,x2,y2 = map(int, person.xyxy[0])
        cx, cy = (x1+x2)//2, (y1+y2)//2
        detections.append((cx, cy, x1, y1, x2, y2))
    
    if not tracks:
        assign_map.clear()
        for cx,cy,x1,y1,x2,y2 in detections:
            gid = next_gid; next_gid += 1
            tracks[gid] = {"last_centroid":(cx,cy),"missed":0}
            assign_map[gid] = (x1,y1,x2,y2)
    
    # 3) มิฉะนั้น → build cost matrix แล้ว Hungarian
    else:
        track_ids = list(tracks.keys())
        N, M = len(track_ids), len(detections)
        cost = np.zeros((N, M), dtype=np.float32)
        for i, gid in enumerate(track_ids):
            tx,ty = tracks[gid]["last_centroid"]
            for j, (cx,cy, *_ ) in enumerate(detections):
                cost[i,j] = np.hypot(tx-cx, ty-cy)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_det = set()
        new_assign = {}
        
        # assign matches under MAX_DIST
        for i, j in zip(row_ind, col_ind):
            if cost[i,j] < MAX_DIST:
                gid = track_ids[i]
                cx,cy,x1,y1,x2,y2 = detections[j]
                new_assign[gid] = (x1,y1,x2,y2)
                tracks[gid]["last_centroid"] = (cx,cy)
                tracks[gid]["missed"] = 0
                matched_det.add(j)

        # unmatched detections → new gid
        for j, (cx,cy,x1,y1,x2,y2) in enumerate(detections):
            if j not in matched_det:
                gid = next_gid; next_gid += 1
                new_assign[gid] = (x1,y1,x2,y2)
                tracks[gid] = {"last_centroid":(cx,cy),"missed":0}

        # unmatched tracks → increase missed
        for gid in track_ids:
            if gid not in new_assign:
                tracks[gid]["missed"] += 1

        # prune old tracks
        for gid in list(tracks):
            if tracks[gid]["missed"] > DROPOUT_THRESHOLD:
                del tracks[gid]
                assign_map.pop(gid, None)
                buffers.pop(gid, None)
                landmark_history.pop(gid, None)
                id_logs.pop(gid, None)
                
                if gid in pose_instances:
                    pose_instances[gid].close()  # <- สำคัญมาก
                    del pose_instances[gid]

        assign_map = new_assign
        
        for gid, (hx1,hy1,hx2,hy2) in assign_map.items():
            # เก็บข้อมูลเบื้องต้นของแต่ละ global_id
            id_logs.setdefault(gid, {
                "actions": [],
                "last_seen_frame": frame_idx,
                "total_frames": 0,
            })
            
            # อัปเดตข้อมูลทุกเฟรมที่เจอ
            id_logs[gid]["last_seen_frame"] = frame_idx
            id_logs[gid]["total_frames"] += 1
            if id_logs[gid]["total_frames"] < MIN_VALID_FRAMES:
                continue
            
            buffers.setdefault(gid, deque(maxlen=SEQUENCE_LENGTH))
            if gid not in pose_instances:
                pose_instances[gid] = mpPose.Pose(
                    static_image_mode=False, model_complexity=1,
                    smooth_landmarks=True, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)
            landmark_history.setdefault(gid, deque(maxlen=SMOOTH_FRAMES))

                        
            cv.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
            
            # Pose ROI
            roi = frame[hy1:hy2, hx1:hx2]
            if roi.size == 0: continue
            roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
            
            pose = pose_instances.get(gid)
            if pose is None:
                continue
            pose_results = pose_instances[gid].process(roi_rgb)
            if not pose_results.pose_landmarks: continue
            lms = pose_results.pose_landmarks.landmark
            collect_pose_landmarks(buffers[gid], lms)
            
            # collect raw full landmarks for smoothing
            raw_pts = np.array([(lm.x, lm.y) for lm in lms])
            landmark_history[gid].append(raw_pts)
            
            # smooth landmarks if available
            if len(landmark_history[gid]) > 0:
                hist = np.stack(landmark_history[gid], axis=0)
                smooth_pts = np.mean(hist, axis=0)  # (33,2)
            else:
                smooth_pts = raw_pts
            
            # draw smoothed skeleton on ROI
            # for (start, end) in mpPose.POSE_CONNECTIONS:
            #     x1n, y1n = smooth_pts[start]
            #     x2n, y2n = smooth_pts[end]
            #     p1 = mpDraw._normalized_to_pixel_coordinates(x1n, y1n, hx2-hx1, hy2-hy1)
            #     p2 = mpDraw._normalized_to_pixel_coordinates(x2n, y2n, hx2-hx1, hy2-hy1)
            #     if p1 and p2:
            #         cv.line(roi, p1, p2, (0, 255, 255), 2)
            # for nx, ny in smooth_pts:
            #     px, py = int(nx*(hx2-hx1)), int(ny*(hy2-hy1))
            #     cv.circle(roi, (px, py), 3, (255, 255, 0), -1)
                
            action_label = "unknown"
            avg = 0
            
            if len(buffers[gid]) == SEQUENCE_LENGTH:
                action_label, avg = get_action(buffers[gid])
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
                            cv.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
                    
                    if matched or iou_val > 0.1:
                        action_label = "carrying"
                        break  # ไม่ต้องเช็คกล่องอื่นแล้ว
                
                id_logs[gid]["actions"].append(action_label)
                
                if action_label == 'carrying':
                    # print("ID: {} | Action: {} | Object type: {}".format(global_id, action_label, object_label))
                    cv.putText(
                    frame,
                    f"ID:{gid} | {hconf:.2f} | Action: {action_label} | Object: {object_label}",
                    (hx1, hy1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )
                else:    
                    # print("ID: {} | Action: {}".format(global_id, action_label))
                    cv.putText(
                        frame,
                        f"ID:{gid} | {hconf:.2f} | Action: {action_label} | Avg: {avg:.2f}",
                        (hx1, hy1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1,
                    )
            else:
                cv.putText(
                        frame,
                        f"ID:{gid} | {hconf:.2f}",
                        (hx1, hy1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1,
                    )
            
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