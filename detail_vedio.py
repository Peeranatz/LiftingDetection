import cv2

video_path = "/Users/balast/Desktop/LiftingProject/LiftingDetection/video_datasets/Carrying/Datatest1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอได้")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    print(f"✅ Video Info: {video_path}")
    print(f"  • Resolution  : {width} x {height}")
    print(f"  • FPS         : {fps}")
    print(f"  • Total Frames: {total_frames}")
    print(f"  • Duration    : {duration:.2f} seconds")

cap.release()
