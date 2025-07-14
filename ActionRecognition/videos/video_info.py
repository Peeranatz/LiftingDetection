#!/usr/bin/env python3
import cv2
import os
import argparse

def print_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] ไม่สามารถเปิดวิดีโอ: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # แปลง FOURCC code เป็น string
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    duration_sec = frame_count / fps if fps else 0

    print(f"--- Info for {os.path.basename(video_path)} ---")
    print(f"  • Resolution  : {width} x {height}")
    print(f"  • FPS         : {fps:.2f}")
    print(f"  • Total frames: {frame_count}")
    print(f"  • Duration    : {duration_sec:.2f} sec ({duration_sec/60:.2f} min)")
    print(f"  • Codec (FOURCC): {fourcc_str}")
    print()

    cap.release()

def main():
    parser = argparse.ArgumentParser(
        description="Print basic info of video file or all videos in a folder"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to video file OR folder containing video files"
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        # input คือโฟลเดอร์ → วนลูปดูทุกไฟล์
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(args.input, fname)
                print_video_info(video_path)

    elif os.path.isfile(args.input):
        # input คือไฟล์เดียว
        print_video_info(args.input)

    else:
        print(f"[ERROR] Invalid input: {args.input}")


if __name__ == "__main__":
    main()

