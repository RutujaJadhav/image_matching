import cv2
imgInput = cv2.imread('C:/Users/Rutuja Jadhav/IMAGES/Desktop/NEW73.jpg',0)

# convert image to grayscale
#imgGray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)         

# invert black and white
newRet, binaryThreshold = cv2.threshold(imgInput,127,255,cv2.THRESH_BINARY_INV)

# dilation
rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,10))

rectdilation = cv2.dilate(binaryThreshold, rectkernel, iterations = 1)

outputImage = imgInput.copy()

npaContours, npaHierarchy = cv2.findContours(rectdilation.copy(),        
                                             cv2.RETR_EXTERNAL,                 
                                             cv2.CHAIN_APPROX_SIMPLE)           

for npaContour in npaContours:                         
    if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          

        [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         

        cv2.rectangle(outputImage,           
              (intX, intY),                 # upper left corner
              (intX+intW,intY+intH),        # lower right corner
              (0, 0, 255),                  # red
              2)                            # thickness

        # Get subimage of word and find contours of that word
        imgROI = binaryThreshold[intY:intY+intH, intX:intX+intW]   


        subContours, subHierarchy = cv2.findContours(imgROI.copy(),        
                                             cv2.RETR_EXTERNAL,                 
                                             cv2.CHAIN_APPROX_SIMPLE) 

        # This part is not working as I am expecting
        for subContour in subContours:

            [pointX, pointY, width, height] = cv2.boundingRect(subContour) 

            cv2.rectangle(outputImage,
                         (intX+pointX, intY+pointY),            
                         (intX+width, intY+height),       
                         (0, 255, 0),
                         2)



cv2.imshow("original", imgInput)
cv2.imshow("rectdilation", rectdilation)
cv2.imshow("threshold", binaryThreshold)
cv2.imshow("outputRect", outputImage)

cv2.waitKey(0);
