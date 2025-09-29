ST-GCN Data Preprocessing Pipeline

1.Collect raw keypoints
Input: วิดีโอ หรือ live cam
ใช้ YOLO Pose → เก็บ (x, y, conf) ของทุก joint 17 จุด ต่อ frame
Group ตาม track_id → เพื่อให้แต่ละคนมี sequence ของตัวเอง

2.Temporal Alignment (Normalize sequence length)
ST-GCN ต้องการให้ทุก sequence มี จำนวนเฟรม (T) เท่ากัน เช่น 30, 60, หรือ 300
วิธีแก้: ถ้าเฟรมน้อยกว่า → pad ด้วยศูนย์ หรือ interpolate หรือ ถ้าเฟรมมากกว่า → crop หรือ sample (uniform sampling)

3.Spatial Normalization
เพื่อไม่ให้โมเดลสับสนระหว่างคนตัวเล็ก/ใหญ่, อยู่ใกล้/ไกลกล้อง:
เลือก root joint เช่น MidHip (หรือเฉลี่ย LeftHip + RightHip) → ย้าย skeleton ทั้งตัวให้ origin = (0,0)
Scale ด้วยความสูง (เช่น Hip ↔ Shoulder distance) → normalize ให้ทุก skeleton มีสเกลใกล้เคียงกัน

4.Reorder joints
ตรวจสอบว่า index joints (0–16) ของ YOLO/COCO ตรงกับ graph ที่ ST-GCN ใช้
เช่น COCO-17 joints → mapping edges

5.Build tensor (N, C, T, V, M)
N = จำนวน sample (วิดีโอ/sequence)
C = channel (2 = x,y หรือ 3 = x,y,conf)
T = temporal length (เฟรม, เช่น 30)
V = จำนวน joints (17)
M = จำนวนคนต่อคลิป (ส่วนใหญ่ใช้ 1)

6.Labeling
กำหนด action class ให้แต่ละ sequence เช่น:
"lifting" → 0, "walking" → 1, "carrying" → 2
เก็บเป็นไฟล์ .json หรือ .pkl

7.Save dataset
บันทึกไฟล์ .npy สำหรับ skeleton แต่ละคลิป
สร้างไฟล์ train.pkl / val.pkl สำหรับ annotation list
