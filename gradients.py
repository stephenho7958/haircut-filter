from pickle import TRUE
import numpy as np
import cv2 as cv
import PyCeres
import matplotlib.pyplot as plt

#read video
cap = cv.VideoCapture('./data/facevid_smallmotion.mp4')
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
cv.imwrite('./data/grayscale.png',old_gray)

#Apply gaussian blur, with 5x5 gaussian kernel- larger kernel = more blur
old_blur = cv.GaussianBlur(old_gray,(11,11),0)
cv.imwrite('./data/blurred11.png',old_blur)

#get gradients using sobel kernel. Faster than manual but may be good to compute it manually
sobelx = cv.Sobel(old_blur,cv.CV_64F,1,0,ksize=5)
sobelx = cv.convertScaleAbs(sobelx)
sobely = cv.Sobel(old_blur,cv.CV_64F,0,1,ksize=5)
sobely = cv.convertScaleAbs(sobely)
sobelcombined = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv.imwrite('./data/sobelx.png',sobelx)
cv.imwrite('./data/sobely.png', sobely)
cv.imwrite('./data/sobelcombined.png',sobelcombined)


zerosX = []
zerosY = []
for rowI, row in enumerate(sobelx):
    colIList = [idx for idx, elem in enumerate(row) if elem==0]
    for colI in colIList:
        zerosX.append([rowI, colI])
        cv.circle(sobelx, (colI, rowI),1,(255,0,0),-1)
print(len(row),len(sobelx))

for rowI, row in enumerate(sobely):
    colIList = [idx for idx, elem in enumerate(row) if elem==0]
    for colI in colIList:
        zerosX.append([rowI, colI])
        cv.circle(sobely, (colI, rowI),1,(255,0,0),-1)
print(len(row),len(sobely))
cv.imwrite('./data/sobelxcir.png',sobelx)
cv.imwrite('./data/sobelycir.png', sobely)
