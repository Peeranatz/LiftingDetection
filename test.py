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

yolo_model = YOLO(
    "/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/best_v3.pt"
)

SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.5

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

last_action = {}  # track_id -> last action label
action_start = {}  # track_id -> datetime of start
buffers = {}  # track_id -> deque

pose_instances = {}  # track_id -> Pose object

# target_id = 1
# debug_buffer = list()

SELECTED_JOINTS = [
    25,
    26,  # left knee, right knee
    27,
    28,  # left ankle, right ankle
    29,
    30,  # left heel, right heel
    31,
    32,  # left foot index, right foot index
]
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture(
#     "/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/data/test_video/test_video_4.mp4"
# )
pTime = 0


def is_hand_in_box(lms, person_box, box, img_frame_w, img_frame_h):
    px1, py1, px2, py2 = person_box
    roi_w = px2 - px1
    roi_h = py2 - py1

    for i in [15, 16]:  # LEFT_WRIST, RIGHT_WRIST
        # แปลง normalized ROI → พิกัด full-frame
        x = int(px1 + lms[i].x * roi_w)
        y = int(py1 + lms[i].y * roi_h)
        # เช็คว่าอยู่ในกล่อง object ไหม
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return True
    return False


def is_box_fully_inside_person(person_box, object_box):
    bx1, by1, bx2, by2 = object_box
    px1, py1, px2, py2 = person_box
    return bx1 >= px1 and by1 >= py1 and bx2 <= px2 and by2 <= py2


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def is_carrying(person_box, object_boxes, iou_threshold: float = 0.03):
    for obj in object_boxes:
        iou = compute_iou(person_box, obj)
        print("IOU:", iou)
        if compute_iou(person_box, obj) > iou_threshold:
            return True

    return False


def collect_pose_landmarks(buffer: deque, landmarks):
    pose_array = []
    for j in SELECTED_JOINTS:
        p = landmarks[j]
        pose_array.append([p.x, p.y])

    buffer.append(np.array(pose_array))


def get_action(buffer: deque, std_threshold: float = 0.022) -> str:
    if len(buffer) < 10:
        return "unknown"

    arr = np.array(buffer)
    stds = np.std(arr, axis=0)

    avg_std = np.mean(stds)
    if avg_std < std_threshold:
        return "standing", avg_std
    else:
        return "moving", avg_std


def expand_bbox(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.1):
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    nx1 = max(0, x1 - pad_w)
    # ny1 = max(0, y1 - pad_h)
    ny1 = y1
    nx2 = min(img_w, x2 + pad_w)
    # ny2 = min(img_h, y2 + pad_h)
    ny2 = y2
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


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    yolo_results = yolo_model.track(
        source=frame, stream=False, tracker="bytetrack.yaml"
    )[0]

    object_boxes = []

    for box in yolo_results.boxes:
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())  # ดึง index ของคลาสที่ตรวจพบเจอ
        label = yolo_model.names[cls]

        if conf < CONF_THRESHOLD:
            continue

        track_id = int(box.id[0]) if box.id is not None else -1
        # print(track_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])

        if label == "box":
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            object_boxes.append((x1, y1, x2, y2))
            cv.putText(
                frame,
                f"ID:{track_id} | {label} {conf:.2f}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                3,
            )
            continue

        if label == "human":
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_box = (x1, y1, x2, y2)
            if track_id not in buffers:
                buffers[track_id] = deque(maxlen=SEQUENCE_LENGTH)

            if track_id not in last_action:
                last_action[track_id] = None
                action_start[track_id] = None

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)

            if track_id not in pose_instances:
                pose_instances[track_id] = mpPose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )
            pose_results = pose_instances[track_id].process(roi_rgb)
            if not pose_results.pose_landmarks:
                continue

            mpDraw.draw_landmarks(
                roi, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS
            )
            h_roi, w_roi, _ = roi.shape
            lm = pose_results.pose_landmarks.landmark
            collect_pose_landmarks(buffers[track_id], lm)

            # if track_id == target_id:
            #     debug_buffer.append(buffers[track_id][-1])

            carring = False
            for obj_box in object_boxes:
                if (
                    is_box_fully_inside_person(person_box, obj_box)
                    or compute_iou(person_box, obj_box) > 0.03
                    or is_hand_in_box(
                        lm, person_box, obj_box, frame.shape[1], frame.shape[0]
                    )
                ):
                    carrying = True
                    break

            if len(buffers[track_id]) == SEQUENCE_LENGTH:
                if carring:
                    action_label = "carrying"
                    # print("Track ID: {} | Action = {}".format(track_id, action_label))
                else:
                    action_label, avg = get_action(buffers[track_id])
                    # print(
                    #     "Track ID: {} | Action = {} | Avg = {}".format(
                    #         track_id, action_label, avg
                    #     )
                    # )

                now = dt.datetime.now()

                if action_label != last_action[track_id]:
                    if last_action[track_id]:
                        log_action(
                            person_id=str(track_id),
                            action=last_action[track_id],
                            start_time=action_start[track_id],
                            end_time=now,
                            object_type=None,  # หรือใส่ label ที่ detect ได้ก็ได้
                        )
                    last_action[track_id] = action_label
                    action_start[track_id] = now

                cv.putText(
                    frame,
                    f"ID:{track_id} | {label} {conf:.2f} | Action: {action_label}",
                    (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3,
                )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(
        frame, f"FPS: {int(fps)}", (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
    )

    cv.imshow("Multi-Person pose", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

# if len(debug_buffer) >= 10:
#     plot_joint_std(debug_buffer)
# else:
#     print("ยังเก็บ buffer ไม่พอจะ plot")
