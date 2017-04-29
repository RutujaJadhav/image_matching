from PIL import Image
import os, sys
import cv2
import numpy as np
##path = 'F:/RESEARCH/WHALE/FinalPackage-whales-SEACLEF2017/FinalPackage-SEACLEF2016'
##dirs = os.listdir( path )
##
##
##for item in dirs:
##    if os.path.exists(path+item):
##        print 'k'
##        im = Image.open(path+item)
##        f, e = os.path.splitext(path+item)
##        imResize = im.resize((200,200), Image.ANTIALIAS)
##        imResize.save(f + ' resized.jpg', 'JPEG', quality=90)


path1 = 'F:/RESEARCH/WHALE/FinalPackage-whales-SEACLEF2017/FinalPackage-SEACLEF2016/'   
path2 = 'F:/RESEARCH/WHALE/FinalPackage-whales-SEACLEF2017/NEW'   

listing = os.listdir(path1)
print listing
for file in listing:
    
    img=cv2.imread(path1 + file)
    height, width = img.shape[:2]
    
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


                                
    cv2.imwrite("NEW"+str(file),final)        


