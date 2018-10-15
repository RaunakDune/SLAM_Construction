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

disp = Display(W,H)
fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (W,H))
    kps, des, matches = fe.extact(img)

    for p in kps:
        u,v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (u,v), color = (0,0,255), radius=3)
    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("truck_driving.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break