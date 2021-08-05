import cv2 as cv
import numpy as np
import imutils

img = cv.imread('CoralReef1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
output = img.copy()

def emptyFunction():
    pass

#color palette
LowImageBlue = np.zeros((512, 512, 3), np.uint8)
HighImageBlue = np.zeros((512, 512, 3), np.uint8)
windowName = "config"
cv.namedWindow(windowName)
cv.namedWindow("Canny")
cv.createTrackbar("HLow", windowName, 0, 179, emptyFunction)
cv.createTrackbar("SLow", windowName, 0, 255, emptyFunction)
cv.createTrackbar("VLow", windowName, 0, 255, emptyFunction)

cv.createTrackbar("HHigh", windowName, 0, 179, emptyFunction)
cv.createTrackbar("SHigh", windowName, 0, 255, emptyFunction)
cv.createTrackbar("VHigh", windowName, 0, 255, emptyFunction)

cv.createTrackbar("CannyLow", "Canny", 0, 255, emptyFunction)
cv.createTrackbar("CannyHigh", "Canny", 0, 255, emptyFunction)

orb = cv.ORB_create()

while True:
    HLow = cv.getTrackbarPos("HLow", windowName)
    SLow = cv.getTrackbarPos("SLow", windowName)
    VLow = cv.getTrackbarPos("VLow", windowName)

    HHigh = cv.getTrackbarPos("HHigh", windowName)
    SHigh = cv.getTrackbarPos("SHigh", windowName)
    VHigh = cv.getTrackbarPos("VHigh", windowName)

    CannyLow = cv.getTrackbarPos("CannyLow", "Canny")
    CannyHigh = cv.getTrackbarPos("CannyHigh", "Canny")

    #LowImageBlue[:] = [HLow, SLow, VLow]
    LowImageBlue[:] = [100, 125, 125]
    LowImageBlue = cv.cvtColor(LowImageBlue, cv.COLOR_HSV2BGR)

    #HighImageBlue[:] = [HHigh, SHigh, VHigh]
    HighImageBlue[:] = [115, 255, 255]
    HighImageBlue = cv.cvtColor(HighImageBlue, cv.COLOR_HSV2BGR)

    if cv.waitKey(1) == 27:
        cv.imshow("LowImageBlue", LowImageBlue)
        cv.imshow("HighImageBlue", HighImageBlue)

        #LowerBlue = np.array([HLow, SLow, VLow])
        #UpperBlue = np.array([HHigh, SHigh, VHigh])
        LowerBlue = np.array([100, 125, 125])
        UpperBlue = np.array([115, 255, 255])

        mask = cv.inRange(hsv, LowerBlue, UpperBlue)
        cv.imshow("mask", mask)

        filter = cv.bitwise_and(gray, gray, mask = mask)
        cv.imshow("filter", filter)

        KeyPoint = orb.detect(filter, None)
        KeyPoint, des = orb.compute(filter, KeyPoint)

        output = cv.drawKeypoints(output, KeyPoint, None, color = (0,255,0), flags = 0)
        cv.imshow("output", output)


        # Contours
        '''
        contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        output = img.copy()

        for c in contours:
            cv.drawContours(output, [c], -1, (255, 0,0), 3)
            cv.imshow('Contours', output)'''


        # Hough Line Detection
        '''rey = cv.bitwise_and(gray, gray, mask = mask)
        BlurGrey = cv.GaussianBlur(grey,(5, 5),0)

        edges = cv.Canny(BlurGrey, CannyLow, CannyHigh)
        cv.imshow("edges", edges)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 400  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img)  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)


        # Draw the lines on the  image
        lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
        cv.imshow("img", line_image)'''
