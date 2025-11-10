import os
import cv2
from visualize import *
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

yolo_model = YOLO("/Users/balast/Desktop/Desktop - All file/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/model/human.pt")

# ===================== [ADDED] NTU-25 drawing config =====================
SHOW_NTU = True          # เปิดหน้าต่างโชว์โครง NTU-25
REPLACE_ANNOTATED = False # True = ใช้ NTU-25 แทนหน้าต่าง "Annotated"

# คู่กระดูก (edges) ของ NTU-25: ใช้ index แบบ 0-based (ตรงกับ array ขนาด 25)
NTU_EDGES = [
    (0,1),(1,2),(2,3),(2,20),
    (0,16),(16,17),(17,18),(18,19),
    (0,12),(12,13),(13,14),(14,15),
    (2,4),(4,5),(5,6),(6,21),(6,22),
    (2,8),(8,9),(9,10),(10,23),(10,24),
]

def _draw_ntu25_on_image(bgr_image, ntu25_xyz, use_pixel=True):
    """
    วาดโครง NTU-25 ลงบนภาพ BGR
    ntu25_xyz: np.array (25,3) ในพิกเซลถ้า use_pixel=True
               ถ้า normalized [0..1] ให้ set use_pixel=False
    """
    h, w = bgr_image.shape[:2]
    pts = ntu25_xyz.copy()

    # ถ้าเป็น normalized → แปลงเป็นพิกเซลก่อนวาด
    if not use_pixel:
        pts[:, 0] = pts[:, 0] * w
        pts[:, 1] = pts[:, 1] * h

    # วาดกระดูก (เส้น)
    for a, b in NTU_EDGES:
        xa, ya = int(pts[a,0]), int(pts[a,1])
        xb, yb = int(pts[b,0]), int(pts[b,1])
        cv2.line(bgr_image, (xa, ya), (xb, yb), (0, 255, 255), 2)

    # วาดจุด
    for i in range(pts.shape[0]):
        x, y = int(pts[i,0]), int(pts[i,1])
        cv2.circle(bgr_image, (x,y), 3, (0, 128, 255), -1)

    return bgr_image
# ========================================================================


# ===================== [ADDED] Configs & Utilities =====================
SAVE_NPY = True        # เผื่ออนาคต: เซฟลำดับ keypoints NTU25 หลังจบลูป
SAVE_PATH = "ntu25_sequence.npy"
USE_PIXEL = True       # True = ค่าพิกเซล, False = normalized [0..1]

def _midpoint(a, b):
    return (a + b) / 2.0

def _safe_stack(points):
    """รวมค่าเฉลี่ยแบบปลอดภัย (มี None ได้)"""
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)

def _extract_mp33_xyz(detection_result, img_w, img_h, use_pixel=True):
    """
    ดึงพิกัด mediapipe 33 จุดเป็น np.array (33,3) หรือ None ถ้าไม่เจอคน
    """
    if not detection_result.pose_landmarks:
        return None
    lms = detection_result.pose_landmarks[0]  # เอาคนหลัก
    pts = []
    for lm in lms:
        if use_pixel:
            x = lm.x * img_w
            y = lm.y * img_h
            z = lm.z * img_w  # scale z ให้สเกลใกล้เคียง x
        else:
            x, y, z = lm.x, lm.y, lm.z
        pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float32)  # (33,3)

def convert_mediapipe33_to_ntu25(mp33):
    """
    แปลง MediaPipe (33,3) -> NTU RGB+D (25,3)
    ใช้จุดเฉลี่ยเพื่อประมาณ Spine/Neck/Hand ให้สอดคล้อง topology ของ NTU
    """
    def P(i):
        if i is None:
            return None
        return mp33[i]

    # จุดหลักจาก MediaPipe
    NOSE = P(0)
    L_SH = P(11); R_SH = P(12)
    L_EL = P(13); R_EL = P(14)
    L_WR = P(15); R_WR = P(16)
    L_PK = P(17); R_PK = P(18)
    L_ID = P(19); R_ID = P(20)
    L_TH = P(21); R_TH = P(22)
    L_HP = P(23); R_HP = P(24)
    L_KN = P(25); R_KN = P(26)
    L_AN = P(27); R_AN = P(28)
    L_HL = P(29); R_HL = P(30)  # เผื่อใช้ต่อยอด
    L_FI = P(31); R_FI = P(32)

    # จุดอนุมาน
    HIP_MID   = _midpoint(L_HP, R_HP) if (L_HP is not None and R_HP is not None) else _safe_stack([L_HP, R_HP])
    SHO_MID   = _midpoint(L_SH, R_SH) if (L_SH is not None and R_SH is not None) else _safe_stack([L_SH, R_SH])
    SPINE_SHO = SHO_MID                            # NTU joint 21
    SPINE_MID = _safe_stack([HIP_MID, SPINE_SHO])  # NTU joint 2
    NECK      = _safe_stack([SPINE_SHO, NOSE])     # NTU joint 3 (คอกลาง)
    HEAD      = NOSE                               # NTU joint 4

    # ฝ่ามือ (NTU ต้องการ 'Hand' ไม่ใช่แค่ข้อมือ) — ใช้จุดเฉลี่ยของฝ่ามือ
    PALM_L = _safe_stack([L_WR, L_ID, L_PK, L_TH])
    PALM_R = _safe_stack([R_WR, R_ID, R_PK, R_TH])

    # เท้า: ใช้ foot index (หน้าเท้า) ถ้าไม่มีให้ fallback เป็นข้อเท้า
    FOOT_L = L_FI if L_FI is not None else L_AN
    FOOT_R = R_FI if R_FI is not None else R_AN

    # เรียงตาม NTU 1..25 (เราเก็บแบบ 0-based index 0..24)
    ntu25 = [
        HIP_MID,      # 1  SpineBase
        SPINE_MID,    # 2  SpineMid
        NECK,         # 3  Neck
        HEAD,         # 4  Head
        L_SH,         # 5  ShoulderLeft
        L_EL,         # 6  ElbowLeft
        L_WR,         # 7  WristLeft
        PALM_L,       # 8  HandLeft (palm center)
        R_SH,         # 9  ShoulderRight
        R_EL,         # 10 ElbowRight
        R_WR,         # 11 WristRight
        PALM_R,       # 12 HandRight (palm center)
        L_HP,         # 13 HipLeft
        L_KN,         # 14 KneeLeft
        L_AN,         # 15 AnkleLeft
        FOOT_L,       # 16 FootLeft
        R_HP,         # 17 HipRight
        R_KN,         # 18 KneeRight
        R_AN,         # 19 AnkleRight
        FOOT_R,       # 20 FootRight
        SPINE_SHO,    # 21 SpineShoulder
        L_ID,         # 22 HandTipLeft  (index tip)
        L_TH,         # 23 ThumbLeft    (thumb tip)
        R_ID,         # 24 HandTipRight (index tip)
        R_TH,         # 25 ThumbRight   (thumb tip)
    ]

    # กัน None: แทนด้วยศูนย์เพื่อไม่ให้พัง (ปรับเป็น mask/np.nan ภายหลังได้)
    ntu25 = [np.zeros(3, dtype=np.float32) if p is None else p for p in ntu25]
    return np.stack(ntu25, axis=0).astype(np.float32)
# ===================== [/ADDED] =====================================

dir_path = os.path.dirname(os.path.abspath(__file__))

lite = "pose_landmarker_lite.task"
heavy = "pose_landmarker_heavy.task"
full = "pose_landmarker_full.task"
model_path = os.path.join(dir_path, full)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Setup Detection Object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

cam = cv2.VideoCapture(
    "/Users/balast/Desktop/Desktop - All file/LiftingProject/LiftingDetection/ActionRecognition/data/test_video/test_video_2.mp4"
)

# ===================== [ADDED] buffer สำหรับลำดับ NTU25 =====================
ntu25_sequence = []   # จะเก็บเป็นรูป (T, 25, 3)
# ============================================================================

current_time = 0
prev_time = 0
# ก่อนลูป: เหมือนเดิม

# ==== YOLO configs (ADD) ====
YOLO_CONF = 0.35       # กรอง box ที่มั่นใจพอ
YOLO_IOU  = 0.45
MAX_PEOPLE = 5         # จำกัดคน/เฟรม (กันช้า)

while True:
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # ✅ แก้จุดที่ 1
    ret, BGR = cam.read()
    if not ret:
        print("Video ended or cannot read frame.")
        break
    BGR_time = BGR.copy()

    # เขียน FPS ก่อน
    cv2.putText(BGR_time, f"FPS: {round(fps, 1)}",
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    # ==== per-frame storage (ADD) ====
    frame_ntu25_list = []   # เก็บ ntu25 ของแต่ละคนในเฟรมนี้
    # ==== YOLO detect persons (REPLACE) ====
    results = yolo_model(BGR, conf=YOLO_CONF, iou=YOLO_IOU, classes=[0], verbose=False)

    # จัดเรียง box ตามความมั่นใจ สูง→ต่ำ (ADD)
    boxes = results[0].boxes
    if boxes is None: boxes = []
    # แปลงเป็น list ของ (score, xyxy, cls)
    det_list = []
    for b in boxes:
        score = float(b.conf[0])
        xyxy  = b.xyxy[0].tolist()
        det_list.append((score, xyxy))
    det_list.sort(key=lambda x: x[0], reverse=True)
    det_list = det_list[:MAX_PEOPLE]

    # ==== loop people (REPLACE) ====
    H, W = BGR.shape[:2]
    for score, (x1, y1, x2, y2) in det_list:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        # clip ขอบ (ADD)
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        if x2 <= x1 or y2 <= y1: 
            continue

        crop = BGR[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        RGB_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_crop)
        result = detector.detect(mp_image)

        h, w = RGB_crop.shape[:2]
        mp33 = _extract_mp33_xyz(result, w, h, use_pixel=USE_PIXEL)
        if mp33 is None:
            continue

        ntu25 = convert_mediapipe33_to_ntu25(mp33)
        # shift กลับสู่พิกัดภาพเต็ม (ADD)
        ntu25[:, 0] += x1
        ntu25[:, 1] += y1

        # เก็บของ "คนนี้" ไว้ในเฟรมนี้ (ADD)
        frame_ntu25_list.append(ntu25)

        # วาด skeleton ของคนนี้ (UNCHANGED)
        _draw_ntu25_on_image(BGR_time, ntu25, use_pixel=True)
        
        # ==== Draw YOLO bounding box and class label (ADD) ====
        # วาดกรอบสี่เหลี่ยม
        cv2.rectangle(BGR_time, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # สร้างข้อความ: class + confidence
        label = f"person {score:.2f}"

        # หาความกว้างของข้อความ เพื่อขยายพื้นหลัง
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # วาดพื้นหลังสีดำข้างบน bbox (กันอ่านยาก)
        cv2.rectangle(BGR_time, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)

        # วาดข้อความ class
        cv2.putText(BGR_time, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        
        
    
    # ==== push per-frame list (ADD) ====
    # เก็บเป็นลิสต์ของ (25,3) ต่อคนในเฟรมนี้; จะจัดรูปตอน save
    ntu25_sequence.append(frame_ntu25_list)
    
    cv2.imshow("YOLO + Pose", BGR_time)

    if cv2.waitKey(1) == ord("q"):
        break


# ===================== [ADDED] cleanup & save =====================
cam.release()
cv2.destroyAllWindows()

# if SAVE_NPY and len(ntu25_sequence) > 0:
#     seq = np.stack(ntu25_sequence, axis=0)  # (T, 25, 3)
#     np.save(SAVE_PATH, seq)
#     print(f"✅ Saved NTU25 sequence: {seq.shape} -> {SAVE_PATH}")
# ================================================================
