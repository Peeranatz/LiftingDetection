import cv2 
import torch
import numpy as np 
import yaml
from collections import deque 
from torchvision import transforms 
from PIL import Image 
from ultralytics import YOLO 

from ActionRecognition.models.cnn_lstm import CNNLSTMActionModel

with open("/Users/balast/Desktop/LiftingProject/LiftingDetection/config.yaml", "r") as f:
    config = yaml.safe_load(f)

VIDEO_SOURCE = config["VIDEO_SOURCE"]
HUMAN_MODEL_PT = config["HUMAN_MODEL_PT"]
SEQ_LEN = config["SEQ_LEN"]
CONF_THRESH = config["CONF_THRESH"]
BATCH_SIZE = config["BATCH_SIZE"]
IMG_SIZE = tuple(config["IMG_SIZE"])
DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")

yolo = YOLO(HUMAN_MODEL_PT)

# CNN-LSTM inference model 
num_classes = 6
model = CNNLSTMActionModel(hidden_size=256, num_classes=num_classes)
model.load_state_dict(torch.load("/Users/balast/Desktop/LiftingProject/LiftingDetection/ActionRecognition/models/cnn_lstm_model.pth",map_location=DEVICE))
model.to(DEVICE).eval()

# Transform for ROI frames
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

frame_buffers = {}

def infer_action(buffer: deque):
    seq = torch.stack(list(buffer), dim=0)      # [T, C, H, W]
    inp = seq.unsqueeze(0).to(DEVICE)           # [1, T, C, H, W]
    with torch.no_grad():
        out = model(inp)                        # [1, num_classes]
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()
        conf = probs[pred].item()
    return pred, conf 

LABELS = {
    0: "carry_heavy",
    1: "carry_normal",
    2: "carry_on_shoulder",
    3: "carry_together",
    4: "pull_box",
    5: "push_box"
}


cap = cv2.VideoCapture(VIDEO_SOURCE)
pTime = cv2.getTickCount()

while True:
    success, frame = cap.read()
    if not success:
        continue 
    
    res = yolo.track(
        frame,
        tracker="/Users/balast/Desktop/LiftingProject/LiftingDetection/HumanBox_Insight_YOLO/tracker/bytetrack.yaml",
        persist=True,
    )[0]
    boxes = res.boxes.xywh.cpu().numpy()
    ids   = res.boxes.id.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    
    for (x, y, w, h), tid, conf in zip(boxes, ids, confs):
        if conf < CONF_THRESH:
            continue 
        
        x1, y1 = int(x-w/2), int(y-h/2)
        x2, y2 = int(x+w/2), int(y+h/2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue 
        
        # To PIL & transform 
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        tensor = transform(pil) # [C, H, W]
        
        # Push to buffer
        buf = frame_buffers.setdefault(tid, deque(maxlen=SEQ_LEN))
        buf.append(tensor)
        
        action_label = "..."
        action_conf = 0.0 
        if len(buf) == SEQ_LEN:
            pred, conf_pred = infer_action(buf)
            action_label, action_conf = LABELS[pred], conf_pred
        
        # Draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"ID:{int(tid)} {action_label} {action_conf:.2f}"
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
    cTime = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (cTime - pTime + 1e-6)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Real-time Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()