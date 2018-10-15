import os
#import pygame
import time
import cv2
from display import Display
import numpy as np

os.environ["PYSDL2_DLL_PATH"] = "D:\\Software\\Python libs"

W = 1920 // 2
H = 1080 // 2

disp = Display(W,H)
orb = cv2.ORB_create()

class FeatureExtractor(object):
    GX = 16 // 2
    GY = 12 // 2

    def __init__(self):
        self.orb = cv2.ORB_create(100)
    
    def extact(self, img):

        # sy = img.shape[0]//self.GY
        # sx = img.shape[1]//self.GX

        # akp = []

        # for ry in range(0, img.shape[0], sy):
        #     for rx in range(0, img.shape[1], sx):
        #         img_chunk = img[ry: ry+sy, rx: rx+sx]
        #         kp = self.orb.detect(img_chunk, None)

        #         for p in kp:
        #             p.pt = (p.pt[0] + rx, p.pt[1] + ry)
        #             akp.append(p)

        # return akp

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3)
        return feats

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (W,H))
    kp = fe.extact(img)

    for p in kp:
        u,v = map(lambda x: int(round(x)), p[0])
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