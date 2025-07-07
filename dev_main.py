import cv2
import numpy as np
import time
import mediapipe as mp
import datetime as dt

from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque, Counter
from Database_system.models.action_model import Action


human_model = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/human.pt"
)
object_model = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/box.pt"
)

# Open the video file
video_path = (
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/videos/action_lifamend5.mp4"
)
# video_path = 1

SEQUENCE_LENGTH = 15
CONF_THRESHOLD = 0.6
SMOOTH_FRAMES = 5
landmark_history = defaultdict(lambda: deque(maxlen=SMOOTH_FRAMES))

last_action = {}  # track_id -> last action label
action_start = {}  # track_id -> datetime of start
buffers = {}  # track_id -> deque

last_object_label = {}
last_object_id = {}
object_id_to_person_ids = defaultdict(lambda: deque(maxlen=50))

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

track_history = defaultdict(lambda: [])
final_results = defaultdict(list)

# สร้าง buffer เก็บ push flag per track
push_history = defaultdict(lambda: deque(maxlen=5))

SELECTED_JOINTS = [
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]
CARRYING_LABELS = {
    0: "carry_normal",
    1: "carry_heavy",
    2: "push_forward",
    3: "pull_backward",
    4: "carry_together",
    5: "carry_on_shoulder",
}


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_features_from_skeleton(landmarks, track_id):
    # joints
    l_shoulder, r_shoulder = landmarks[11], landmarks[12]
    l_elbow, r_elbow = landmarks[13], landmarks[14]
    l_wrist, r_wrist = landmarks[15], landmarks[16]
    l_hip, r_hip = landmarks[23], landmarks[24]
    nose = landmarks[0]

    # 1. elbow angles
    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    avg_elbow = (left_elbow_angle + right_elbow_angle) / 2

    # 2. hand height (Y)
    avg_hand_y = (l_wrist.y + r_wrist.y) / 2
    avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    avg_hip_y = (l_hip.y + r_hip.y) / 2
    nose_y = nose.y

    # 3. hand forward/backward (X)
    avg_hand_x = (l_wrist.x + r_wrist.x) / 2
    avg_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2

    print("[DEBUG - PULL CHECK]")
    print(f"  Left angle: {calculate_angle(l_shoulder, l_elbow, l_wrist):.2f}")
    print(f"  Right angle: {calculate_angle(r_shoulder, r_elbow, r_wrist):.2f}")
    print(
        f"  L wrist (x={l_wrist.x:.2f}, y={l_wrist.y:.2f}) vs Shoulder (x={l_shoulder.x:.2f}, y={l_shoulder.y:.2f})"
    )
    print(
        f"  R wrist (x={r_wrist.x:.2f}, y={r_wrist.y:.2f}) vs Shoulder (x={r_shoulder.x:.2f}, y={r_shoulder.y:.2f})"
    )
    print(f"    AVG Hand (x={avg_hand_x:.2f}, y={avg_hand_y:.2f})")
    print(f"    AVG Shoulder (x={avg_shoulder_x:.2f}, y={avg_shoulder_y:.2f})")
    print(f"    AVG Hip ({avg_hip_y:.2f})")
    print(f"    AVG Elbow ({avg_elbow:.2f})")

    print("-----------------------------------")

    is_push = False
    if (
        abs(avg_hand_x - avg_shoulder_x) > 0.05
        and (avg_shoulder_y + 0.02) < avg_hand_y < (avg_hip_y - 0.02)
        and avg_elbow > 100
    ):
        is_push = True

    push_history[track_id].append(is_push)

    left_pull = (
        100 <= calculate_angle(l_shoulder, l_elbow, l_wrist) <= 160
        and l_shoulder.y + 0.02 < l_wrist.y < l_hip.y - 0.02
        and l_wrist.x < l_shoulder.x - 0.03
    )

    right_pull = (
        100 <= calculate_angle(r_shoulder, r_elbow, r_wrist) <= 160
        and r_shoulder.y + 0.02 < r_wrist.y < r_hip.y - 0.02
        and r_wrist.x < r_shoulder.x - 0.03
    )

    if (
        95 <= avg_elbow <= 160
        and avg_shoulder_y + 0.02 < avg_hand_y < avg_hip_y - 0.02
        and ((l_wrist.x > l_shoulder.x + 0.03) or (r_wrist.x > r_shoulder.x + 0.03))
    ):
        return "pull_backward"

    elif sum(push_history[track_id]) >= 2:
        return "push_forward"

    elif (
        100 < avg_elbow <= 165 and avg_shoulder_y + 0.05 < avg_hand_y < avg_hip_y + 0.05
    ):
        return "carry_normal"

    elif avg_elbow > 165 and avg_hand_y > avg_hip_y - 0.03:
        return "carry_heavy"

    elif avg_hand_y > avg_shoulder_y - 0.05 and avg_elbow < 70:
        return "carry_on_shoulder"


def collect_pose_landmarks(buffer: deque, landmarks):
    pose_array = []
    for j in SELECTED_JOINTS:
        p = landmarks[j]
        pose_array.append([p.x, p.y])

    buffer.append(np.array(pose_array))


def point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    result = x1 <= x <= x2 and y1 <= y <= y2
    # print(f"[DEBUG] Point ({x},{y}) in BBox ({x1},{y1})-({x2},{y2})? {result}")
    return result


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_action(buffer: deque, std_threshold: float = 0.015) -> str:
    if len(buffer) < 10:
        return "unknown"

    arr = np.array(buffer)
    avg_std = np.mean(np.std(arr, axis=0))

    return ("idle" if avg_std < std_threshold else "moving"), avg_std


def log_action(
    person_id, action, start_time, end_time, object_type=None, object_id=None
):
    act = Action(
        person_id=person_id,
        action=action,
        object_type=object_type,
        object_id=object_id,  # เพิ่ม object_id
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    act.save()
    print("✅ Logged Action:", person_id, action, object_type, object_id)


cap = cv2.VideoCapture(video_path)
pTime = 0
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_idx += 1
    human_res = human_model.track(
        frame,
        tracker="/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/tracker/bytetrack.yaml",
        persist=True,
    )[0]
    object_res = object_model.track(
        frame,
        tracker="/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/tracker/bytetrack.yaml",
        persist=True,
    )[0]

    if human_res.boxes and human_res.boxes.id is not None:
        human_boxes = human_res.boxes.xywh.cpu()
        track_human_ids = human_res.boxes.id.int().cpu().tolist()
        hconfs = human_res.boxes.conf.cpu().tolist()
        hlabel = human_res.boxes.cls.int().cpu().tolist()
        hnames = human_res.names

        human_centers = {}

        # ลบ buffer ของ ID ที่หายไป
        for tid in list(buffers.keys()):
            if tid not in track_human_ids:
                buffers.pop(tid, None)
                last_action.pop(tid, None)
                action_start.pop(tid, None)
                landmark_history.pop(tid, None)

        # Visualize the result on the frame
        for hbox, htrack_id, hconf, cls_id in zip(
            human_boxes, track_human_ids, hconfs, hlabel
        ):
            if hconf < CONF_THRESHOLD:
                continue
            x, y, w, h = hbox
            hx1, hy1, hx2, hy2 = (
                int(x - w / 2),
                int(y - h / 2),
                int(x + w / 2),
                int(y + h / 2),
            )
            human_label = hnames[cls_id] if hnames else f"class_{cls_id}"
            human_centers[htrack_id] = get_center((hx1, hy1, hx2, hy2))

            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

            # เตรียม buffer ถ้ายังไม่มี
            buffers.setdefault(htrack_id, deque(maxlen=SEQUENCE_LENGTH))
            last_action.setdefault(htrack_id, None)
            action_start.setdefault(htrack_id, None)

            # Pose ROI
            roi = frame[hy1:hy2, hx1:hx2]
            if roi.size == 0:
                continue
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(roi_rgb)
            if not pose_results.pose_landmarks:
                continue

            lms = pose_results.pose_landmarks.landmark
            collect_pose_landmarks(buffers[htrack_id], lms)

            # SKELETON
            pts = np.array([[p.x, p.y] for p in lms])  # shape (33,2)
            landmark_history[htrack_id].append(pts)

            # Compute smoothed landmarks
            hist = np.stack(landmark_history[htrack_id], axis=0)  # (F,33,2)
            smooth_pts = hist.mean(axis=0)

            # วาดด้วย smooth_pts แทน raw lm
            for s, e in mpPose.POSE_CONNECTIONS:
                p1 = mpDraw._normalized_to_pixel_coordinates(
                    smooth_pts[s][0], smooth_pts[s][1], w, h
                )
                p2 = mpDraw._normalized_to_pixel_coordinates(
                    smooth_pts[e][0], smooth_pts[e][1], w, h
                )
                if p1 and p2:
                    pt1 = tuple(map(int, p1))
                    pt2 = tuple(map(int, p2))
                    cv2.line(roi, pt1, pt2, (0, 255, 255), 2)
            for x_n, y_n in smooth_pts:
                px, py = int(x_n * w), int(y_n * h)
                cv2.circle(roi, (px, py), 3, (255, 255, 0), -1)

            action_label = "unknown"
            avg = 0

            if len(buffers[htrack_id]) == SEQUENCE_LENGTH:
                action_label, avg = get_action(buffers[htrack_id])
                if object_res.boxes and object_res.boxes.id is not None:
                    object_boxes = object_res.boxes.xywh.cpu()
                    track_object_ids = object_res.boxes.id.int().cpu().tolist()
                    oconfs = object_res.boxes.conf.cpu().tolist()
                    olabel = object_res.boxes.cls.int().cpu().tolist()
                    onames = object_res.names

                    matched_object_label = None
                    matched_object_id = None

                    for obox, otrack_id, oconf, cls_id in zip(
                        object_boxes, track_object_ids, oconfs, olabel
                    ):
                        if oconf < CONF_THRESHOLD:
                            continue
                        bx, by, bw, bh = obox
                        ox1, oy1, ox2, oy2 = (
                            int(bx - bw / 2),
                            int(by - bh / 2),
                            int(bx + bw / 2),
                            int(by + bh / 2),
                        )

                        object_label = onames[cls_id] if onames else f"class_{cls_id}"
                        object_center = get_center((ox1, oy1, ox2, oy2))

                        # closest_human_id = min(
                        #     human_centers,
                        #     key=lambda pid: np.linalg.norm(np.array(human_centers[pid]) - np.array(object_center))
                        # )

                        matched_human_centers = []
                        matched_human_ids = []

                        # เตรียมจุดของกล่อง
                        points = [
                            ((ox1 + ox2) // 2, (oy1 + oy2) // 2),  # center
                            (ox1, oy1),  # top-left
                            (ox2, oy1),  # top-right
                            (ox1, oy2),  # bottom-left
                            (ox2, oy2),  # bottom-right
                            ((ox1 + ox2) // 2, oy1),  # top-center
                            ((ox1 + ox2) // 2, oy2),  # bottom-center
                            (ox1, (oy1 + oy2) // 2),  # middle-left
                            (ox2, (oy1 + oy2) // 2),  # middle-right
                        ]

                        matched = any(
                            point_in_bbox(px, py, (hx1, hy1, hx2, hy2))
                            for (px, py) in points
                        )

                        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            f"ID:{otrack_id} | {object_label} | {oconf:.2f}",
                            (ox1, oy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 0, 0),
                            1,
                        )

                        # if matched and closest_human_id == htrack_id:
                        if matched:
                            matched_human_centers.append(human_centers[htrack_id])
                            matched_human_ids.append(htrack_id)
                            matched_object_id = str(otrack_id)  # ต้องเป็น str เพื่อบันทึกได้
                            object_id_to_person_ids[matched_object_id].append(htrack_id)
                            most_common_id, freq = Counter(
                                object_id_to_person_ids[matched_object_id]
                            ).most_common(1)[0]

                            deque_vals = list(
                                object_id_to_person_ids[matched_object_id]
                            )
                            counts = Counter(deque_vals)

                            # หาคนที่ผ่าน threshold ทุกคน
                            carriers = [pid for pid, cnt in counts.items() if cnt >= 15]

                            if carriers:
                                # ถ้ามี 2 คนขึ้นไป ให้เป็น carry_together
                                if len(carriers) > 1:
                                    detailed_label = "carry_together"
                                else:
                                    # คนเดียว ก็ไปเช็ค skeleton ตามเดิม
                                    detailed_label = extract_features_from_skeleton(
                                        lms, htrack_id
                                    )

                                action_label = (
                                    detailed_label if detailed_label else "carrying"
                                )
                                matched_object_label = object_label
                                break

                    now = dt.datetime.now()
                    # เก็บผลลัพธ์เฉพาะเมื่อ action เปลี่ยน หรือยังไม่เคยมีมาก่อน
                    if last_action[htrack_id] != action_label:
                        # if last_action[htrack_id]:
                        #     log_action(
                        #         person_id=str(htrack_id),
                        #         action=last_action[htrack_id],
                        #         object_type=last_object_label.get(htrack_id),
                        #         object_id=last_object_id.get(htrack_id),
                        #         start_time=action_start[htrack_id],
                        #         end_time=now
                        #         # หรือใส่ label ที่ detect ได้ก็ได้
                        #     )

                        last_action[htrack_id] = action_label
                        action_start[htrack_id] = now
                        last_object_label[htrack_id] = matched_object_label
                        last_object_id[htrack_id] = matched_object_id
                    # final_results[htrack_id].append((action_label, frame_idx))

                print("Action: {}".format(action_label))

                if action_label == "carrying":
                    cv2.putText(
                        frame,
                        f"ID:{htrack_id} | {hconf:.2f} | Action: {action_label} | Object: {object_label}",
                        (hx1, hy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"ID:{htrack_id} | {hconf:.2f} | Action: {action_label} | Avg: {avg:.2f}",
                        (hx1, hy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1,
                    )

            else:
                cv2.putText(
                    frame,
                    f"ID:{htrack_id} | {hconf:.2f}",
                    (hx1, hy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )

        # Plot the tracks
        # for box, track_id in zip(human_boxes, track_human_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 30 tracks for 30 frames
        #         track.pop(0)

        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        frame,
        f"FPS: {int(fps)} | FRAME: {int(frame_idx)}",
        (70, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 255),
        3,
    )

    cv2.imshow("YOLO11 Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\n=== Action Summary per Person ===")
for pid in sorted(last_action.keys()):
    action = last_action.get(pid, "unknown")
    obj_label = last_object_label.get(pid, None)
    obj_id = last_object_id.get(pid, None)
    start = action_start.get(pid, None)
    print(f"Person ID {pid}:")
    print(f"  • Last action     : {action}")
    print(f"  • Matched object  : {obj_label} (ID={obj_id})")
    print(f"  • Action started  : {start}")
    print()

print("=== Object ↔ Person Matches (last frames) ===")
for obj_id, pid_deque in object_id_to_person_ids.items():
    persons = list(pid_deque)
    counts = Counter(persons)
    print(f"Object ID {obj_id}:")
    print(f"  • Matched person IDs (history): {persons}")
    print(f"  • Frequency count            : {counts}")
    print()

# for obj_id, pid_deque in object_id_to_person_ids.items():
#     print(f"Object ID: {obj_id} → Matched Person IDs (last {pid_deque.maxlen} frames): {max(list(pid_deque))}")


# พิมพ์สรุปผลลัพธ์ทั้งหมดหลังจบการทำงาน
# for track_id, actions in final_results.items():
#     print(f"\n🧍 Person ID: {track_id}")
#     for act, f_idx in actions:
#         print(f"  🕐 Frame {f_idx}: {act}")
