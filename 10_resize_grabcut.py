#resized images
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

img = cv2.imread('C:/Users/Rutuja Jadhav/Desktop/IMAGES/NEW110.jpg')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
height, width = img.shape[:2]
print height,width

#img = cv2.resize(img, (int(width*0.25),int(height*0.25))) 


copy_img=img.copy()
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
a=width-20
b=height-20
##print a,b
rect = (10,10,a,b)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = img*mask2[:,:,np.newaxis]

background=img1-img
background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]
final = background + img1
plt.imshow(img),plt.show()
#cv2.rectangle(final,(20,20),(650,300),(0,255,0),3)
##shifted=cv2.pyrMeanShiftFiltering(final,31,155)
##
##gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
##thresh = cv2.threshold(gray, 0, 255,
##	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

##################
#image = imutils.resize(final, width=min(400, final.shape[1]))

#######################
    
##ret,thresh1 = cv2.threshold(final,127,255,cv2.THRESH_BINARY)
##ret,thresh2 = cv2.threshold(final,127,255,cv2.THRESH_BINARY_INV)
##ret,thresh3 = cv2.threshold(final,127,255,cv2.THRESH_TRUNC)
##ret,thresh4 = cv2.threshold(final,127,255,cv2.THRESH_TOZERO)
##ret,thresh5 = cv2.threshold(final,127,255,cv2.THRESH_TOZERO_INV)
##titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
##images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
##for i in xrange(6):
##    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
##    plt.title(titles[i])
##    plt.xticks([]),plt.yticks([])
##plt.show()



#cv2.rectangle(img,(10,10),(a,b),(0,255,0),3)
##plt.subplot(1,2,1),plt.imshow(copy_img)
##
##plt.subplot(1,2,2),plt.imshow(final)
##plt.show()

