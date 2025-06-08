import cv2 as cv 
import mediapipe as mp 
import time 

mpPose = mp.solutions.pose 
pose = mpPose.pose 

cap = cv.VideoCapture(1)

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break