import cv2 as cv 
import mediapipe as mp 
import time 
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose 
pose = mpPose.Pose()
joints = mpPose.PoseLandmark

NTU_SHOULDER_WIDTH_M = 0.312

cap = cv.VideoCapture(1)

pTime = 0

selected_mediapipe_joints = [
    14, 16, 20, 22, 18,                 # Right arm
    13, 15, 19, 17, 21,                 # Left arm
    24, 26, 28, 23, 25, 27,             # Legs
    "center_shoulder", 0, "center_hip"  # Body 
]

def get_midpoint(lm1, lm2):
    return (lm1.x + lm2.x) / 2, (lm1.y + lm2.y) / 2, (lm1.z + lm2.z) / 2

def normalize_mediapipe_to_ntu(landmarks, image_width, image_height):
    # 1. คำนวณ scale (meter-per-pixel) จากความกว้างระหว่างหัวไหล่
    l_shoulder = landmarks[joints.LEFT_SHOULDER.value]
    r_shoulder = landmarks[joints.RIGHT_SHOULDER.value]
    dx = (l_shoulder.x - r_shoulder.x) * image_width
    dy = (l_shoulder.y - r_shoulder.y) * image_height
    shoulder_pixel_dist = np.hypot(dx, dy)
    m_per_pixel = NTU_SHOULDER_WIDTH_M / shoulder_pixel_dist if shoulder_pixel_dist else 1.0

    # 2. หา center ของสะโพกไว้ใช้เป็น origin
    l_hip = landmarks[joints.LEFT_HIP.value]
    r_hip = landmarks[joints.RIGHT_HIP.value]
    center_hip = get_midpoint(l_hip, r_hip)

    # 3. คำนวณ joint ทั้งหมดที่เลือก
    joint_list = []
    for j in selected_mediapipe_joints:
        if isinstance(j, int):
            p = landmarks[j]
        elif j == "center_shoulder":
            p = type("Point", (), {})()  # สร้าง dummy object
            p.x, p.y, p.z = get_midpoint(l_shoulder, r_shoulder)
        elif j == "center_hip":
            p = type("Point", (), {})()
            p.x, p.y, p.z = center_hip

        # แปลงพิกัด (normalized → pixel → meter → centered → invert y)
        x_px = p.x * image_width
        y_px = p.y * image_height
        z_px = p.z * image_width  # assume z ~ x scale

        cx_px = center_hip[0] * image_width
        cy_px = center_hip[1] * image_height
        cz_px = 0  # ถือว่า hip z = 0

        x_m = (x_px - cx_px) * m_per_pixel
        y_m = -(y_px - cy_px) * m_per_pixel  # invert แกน y
        z_m = (z_px - 0) * m_per_pixel

        joint_list.append([x_m, y_m, z_m])

    return np.array(joint_list), m_per_pixel  # shape (19, 3)

while True:
    success, img = cap.read()
    # img = cv.flip(img, 1)
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        h, w, c = img.shape
        landmarks = results.pose_landmarks.landmark

        # 🔹 เรียก normalize function
        ntu_pose, m_per_pixel = normalize_mediapipe_to_ntu(landmarks, w, h)

        # 🔸 Print ค่าทุก joint ที่ normalize แล้ว
        print("\nNormalized to NTU format (x, y, z) in meters:")
        for i, (x, y, z) in enumerate(ntu_pose):
            print(f"Joint {i:2d}: x={x:.3f}, y={y:.3f}, z={z:.3f}")

        # 🔸 วาดจุดเฉพาะที่ normalize แล้ว (optional)
        for i, (x, y, _) in enumerate(ntu_pose):
            cx = int((x / m_per_pixel) + (landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x +
                                        landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x) / 2 * w)
            cy = int((-(y / m_per_pixel)) + (landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y +
                                            landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y) / 2 * h)
            cv.circle(img, (cx, cy), 4, (0, 255, 0), cv.FILLED)
            
    # if results.pose_landmarks:
    #     mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    #     for id, lm in enumerate(results.pose_landmarks.landmark):
    #         h, w, c = img.shape
    #         print(id, lm)
    #         cx, cy = int(lm.x * w), int(lm.y * h)
    #         cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f"FPS: {int(fps)}", (20, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break