from ultralytics import YOLO
import yaml
import cv2
import time

with open("/Users/balast/Desktop/Desktop - All file/LiftingProject/LiftingDetection/ActionRecognition/GNN/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
MODEL_PATH = config['HUMAN_MODEL_PT']
VIDEO_PATH = config['VIDEO_SOURCE']

KEYPOINT_NAMES = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

# local ID mapping
local_id_map = {} # {track_id: local_id}
next_local_id = 1

if not cap.isOpened():
    print("Failed to open video: {}".format(VIDEO_PATH))
    exit()
    
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        print("Video finished or frame error.")
        break 
    
    frame_count += 1
    
    # คำนวน FPS 
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    
    results = model.track(source=frame, task="pose", persist=True, tracker="bytetrack.yaml", verbose=False)
    result = results[0]
    
    frame_clean = result.orig_img.copy()     # เฟรมดิบ
    frame_auto_annotated = result.plot()     # YOLO วาดให้เรียบร้อย
    
    cv2.putText(frame_clean, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print(f"Frame: {frame_count}", end='\r')
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id]
        
        if class_name == "item":
            bbox = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width  = x2 - x1
            height = y2 - y1
            area   = width * height
            
            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # extract track ID
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1 
            
            if track_id not in local_id_map:
                local_id_map[track_id] = next_local_id
                next_local_id += 1 
            local_id = local_id_map[track_id]
            
            print(f"[{local_id}] BBox: ({x1},{y1},{x2},{y2}) "
            f"→ W={width}px, H={height}px, Area={area}px²")
            
            label = f"ID {local_id}"
            print(f"[{local_id}] PERSON -> BBox: {bbox}, Conf: {conf}")

            cv2.putText(frame_clean, f"ID: {local_id} - Class: {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[i]
                # print("Keypoints (x,y):")
                for idx, (x, y) in enumerate(keypoints):
                    cv2.circle(frame_clean, (int(x), int(y)), 3, (0, 0, 255), -1)
                    name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"Point{idx}"
                    # print(f"{name}: ({x:.1f}, {y:.1f})")

    
    cv2.imshow("YOLOv11 Pose Detection", frame_clean)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit by user.")
        break 

cap.release()
cv2.destroyAllWindows()
