from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
path1 = 'F:/RESEARCH/WHALE/MATCHING_TRIAL DATASET/PNGS/'   
path2 = 'F:/RESEARCH/WHALE/MATCHING_TRIAL DATASET/PNGS2/'

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out



listing_train = os.listdir(path1)
listing_test=os.listdir(path2)
for file1 in listing_train:
    matchmaxlenORB=10
    matchmaxlenSURF=5
    matchmaxlenSIFT=5
    matchfileSIFT=""
    matchfileORB=""
    matchfileSURF=""
    img1=cv2.imread(path1 + file1)
    
    for file2 in listing_test:
        if(file2!=file1):
            img2=cv2.imread(path2+file2)
            orb = cv2.ORB()
            surf=cv2.SURF()
            sift=cv2.SIFT()

            
            kp1_ORB, des1_ORB = orb.detectAndCompute(img1,None)
            kp2_ORB, des2_ORB = orb.detectAndCompute(img2,None)
            bf_ORB = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            matches_ORB = bf_ORB.match(des2_ORB,des1_ORB)
            matches_ORB = sorted(matches_ORB, key = lambda x:x.distance)
            no_of_matches_ORB=len(matches_ORB)
            #print (file1,file2,no_of_matches_ORB)
            if no_of_matches_ORB>matchmaxlenORB:
                matchmaxlenORB=no_of_matches_ORB
                matchfileORB=file2

                
            kp1_SURF, des1_SURF = surf.detectAndCompute(img2,None)
            kp2_SURF, des2_SURF = surf.detectAndCompute(img1,None)
            bf_SURF = cv2.BFMatcher()
            matches_SURF = bf_SURF.knnMatch(des2_SURF,des1_SURF,k=2)
            good_SURF = []
            for m,n in matches_SURF:
                if m.distance < 0.75*n.distance:
                    good_SURF.append([m])
           # matches_SURF= sorted(matches_SURF, key = lambda x:x.distance)
            no_of_matches_SURF=len(good_SURF)
           # print (file1,file2,no_of_matches_SURF)
            if no_of_matches_SURF>matchmaxlenSURF:
                matchmaxlenSURF=no_of_matches_SURF
                matchfileSURF=file2
           
            kp1_SIFT, des1_SIFT = sift.detectAndCompute(img2,None)
            kp2_SIFT, des2_SIFT = sift.detectAndCompute(img1,None)
            bf_SIFT = cv2.BFMatcher()
            matches_SIFT = bf_SIFT.knnMatch(des2_SIFT,des1_SIFT,k=2)
            good_SIFT = []
            for m,n in matches_SIFT:
                if m.distance < 0.75*n.distance:
                    good_SIFT.append([m])
            #print (file1,file2,len(good_SIFT))
            #matches_SIFT= sorted(matches_SIFT, key = lambda x:x.distance)
            no_of_matches_SIFT=len(good_SIFT)
            if no_of_matches_SIFT>matchmaxlenSIFT:
                matchmaxlenSIFT=no_of_matches_SIFT
                matchfileSIFT=file2
            
    #plt.imshow(img3),plt.show()
    print 
    print "ORB " +str(file1)+" "+str(matchfileORB)+" "+str(matchmaxlenORB)
  #  print "SURF"            
  #  print(file1,matchfileSURF,matchmaxlenSURF)
    print "SIFT " +str(file1)+" "+str(matchfileSIFT)+" "+str(matchmaxlenSIFT)
    print "SURF " +str(file1)+" "+str(matchfileSURF)+" "+str(matchmaxlenSURF)
    









    
####print a,b
##    rect = (10,10,a,b)
##    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
##    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
##    img1 = img*mask2[:,:,np.newaxis]
##    background=img1-img
##    background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]
##    final = background + img1
##
##
##                                
##    cv2.imwrite("NEW"+str(file),final)        


