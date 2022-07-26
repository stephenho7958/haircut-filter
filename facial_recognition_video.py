import numpy as np
import cv2 as cv
#load pre-trained model for fromtal face detection from OpenCV. Using Haar Cascades (i.e. Viola-Jones)
classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0

cap = cv.VideoCapture(0)
#print error message if webcam is unavailable
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    count+=1
    #capture frame by frame. ret is True is frame is read correctly, while frame is the image that is the frame.
    ret, frame = cap.read()
    #check frame read correctly
    if not ret:
        print("cannot receive frame. Exiting ...")
        break
    #convert to grayscale. OpenCV uses BGR instead of RGB
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #detect face every 5 loops
    if(count%5==0 or count == 1):
        bboxes = classifier.detectMultiScale(gray)
        # print bounding box for each detected face
    #show rectangle for continuity
    for box in bboxes:
        # extract coordinates
        print(box)
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        cv.rectangle(gray, (x, y), (x2, y2), (0,0,255), 3)

    #show video
    cv.imshow('frame',gray)
    #if user presses q, breaks out of while loop and closes window, stops video capture
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
