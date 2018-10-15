import os
#import pygame
import time
import cv2
from display import Display
import numpy as np

os.environ["PYSDL2_DLL_PATH"] = "D:\\Software\\Python libs"

class FeatureExtractor(object):
    # GX = 16 // 2
    # GY = 12 // 2

    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
    
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
        kps = [cv2.KeyPoint(x = f[0][0], y = f[0][1], _size = 20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        ret = []

        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k = 2)

            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    ret.append((kps[m.queryIdx], self.last['kps'][m.trainIdx]))

        self.last = {'kps': kps, 'des': des}

        # return kps, des, matches
        return ret