import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('F:/RESEARCH/WHALE/MATCHING_TRIAL DATASET/PNGS/NEW14.png',0)
sift=cv2.SIFT()

kp,des=sift.detectAndCompute(img,None)
img2=cv2.drawKeypoints(img,kp,None,(255,0,0))
print len(kp)
print des
plt.imshow(img2)
plt.show()
