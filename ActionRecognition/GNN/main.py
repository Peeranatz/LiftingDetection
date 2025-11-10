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
    "/Users/balast/Desktop/Desktop - All file/LiftingProject/LiftingDetection/ActionRecognition/data/test_video/test_video_3.mp4"
)

# ===================== [ADDED] buffer สำหรับลำดับ NTU25 =====================
ntu25_sequence = []   # จะเก็บเป็นรูป (T, 25, 3)
# ============================================================================

current_time = 0
prev_time = 0
# ก่อนลูป: เหมือนเดิม

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

    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB)
    result = detector.detect(mp_image)

    # แปลง 33 -> 25 (มีอยู่แล้ว)
    h, w = RGB.shape[:2]
    mp33 = _extract_mp33_xyz(result, w, h, use_pixel=USE_PIXEL)
    if mp33 is not None:
        ntu25 = convert_mediapipe33_to_ntu25(mp33)
        ntu25_sequence.append(ntu25)

        # counter เฟรม
        cv2.putText(BGR_time, f"NTU25 frames: {len(ntu25_sequence)}",
                    (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # วาด NTU-25
        if SHOW_NTU:
            if REPLACE_ANNOTATED:
                ntu_vis = BGR.copy()
                _draw_ntu25_on_image(ntu_vis, ntu25, use_pixel=USE_PIXEL)
                cv2.imshow("Annotated", ntu_vis)
            else:
                ntu_vis = BGR.copy()
                _draw_ntu25_on_image(ntu_vis, ntu25, use_pixel=USE_PIXEL)
                cv2.imshow("NTU-25", ntu_vis)

    # ✅ แก้จุดที่ 2: อย่าเขียนทับ Annotated ถ้า REPLACE_ANNOTATED=True
    if not (SHOW_NTU and REPLACE_ANNOTATED):
        annotated = draw_landmarks_on_image(RGB, result)
        cv2.imshow("Annotated", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    # ✅ แก้จุดที่ 3: มาโชว์ Original Frame ตรงนี้ หลังใส่ข้อความครบ
    cv2.imshow("Original Frame", BGR_time)

    if cv2.waitKey(1) == ord("q"):
        break


# ===================== [ADDED] cleanup & save =====================
cam.release()
cv2.destroyAllWindows()

if SAVE_NPY and len(ntu25_sequence) > 0:
    seq = np.stack(ntu25_sequence, axis=0)  # (T, 25, 3)
    np.save(SAVE_PATH, seq)
    print(f"✅ Saved NTU25 sequence: {seq.shape} -> {SAVE_PATH}")
# ================================================================
