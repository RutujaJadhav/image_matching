import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('F:/RESEARCH/WHALE/FinalPackage-whales-SEACLEF2017/FinalPackage-SEACLEF2016/25.jpg',0)
#img=cv2.imread('C:\Users\Rutuja Jadhav\Desktop\if.jpg',0)
#img=('C:\Users\Rutuja Jadhav\Desktop\3.jpg')
#nparr=np.fromstring(img,np.uint8)
#np.reshape(nparr,(3,10))
surf=cv2.SURF(2500)
kp,des=surf.detectAndCompute(img,None)
img2=cv2.drawKeypoints(img,kp,None,(255,0,0),4)
print len(kp)
plt.imshow(img2)
plt.show()
