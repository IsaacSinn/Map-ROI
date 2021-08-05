import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CoralReef1.jpg',0)
cv.imshow("img", img)

# Initiate ORB detector
orb = cv.ORB_create()
kp = orb.detect(img,None)
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
