import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('F:/RESEARCH/WHALE/FinalPackage-whales-SEACLEF2017/FinalPackage-SEACLEF2016/10.jpg',0)
sift=cv2.SIFT()

kp,des=sift.detectAndCompute(img,None)
img2=cv2.drawKeypoints(img,kp)
print len(kp)
print des
plt.imshow(img2)
plt.show()
