import os
#import pygame
import time
import cv2
from display import Display
from featureextractor import FeatureExtractor
import numpy as np

os.environ["PYSDL2_DLL_PATH"] = "D:\\Software\\Python libs"

W = 1920 // 2
H = 1080 // 2
F = 1510

disp = Display(W,H)

K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

fe = FeatureExtractor(K)

def process_frame(img):
    img = cv2.resize(img, (W,H))
    matches = fe.extact(img)

    for pt1, pt2 in matches:
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1,v1), color = (0,0,255), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color = (0,255,0))
    

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("truck_driving.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break