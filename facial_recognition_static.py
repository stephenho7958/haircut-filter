import cv2 as cv
#load pre-trained model for fromtal face detection from OpenCV. Using Haar Cascades (i.e. Viola-Jones)
classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
#read picture
pixels = cv.imread('test1.jpg')
# perform face detection using .detectMultiScale(image,scaleFactor,minNeighbours)
#scaleFactor - how input image is scaled before detection, between 1.05 to 1.40, default is 1.10. For large photos, lower scaleFactor helps, while for small photos, higher scaleFactor is better.
#minNeighbours - how robust the detection is for the face to be reported (number of candidate rectangles found on face). Default is 3, minimum is 1. Higher means needs more rectangles, lower usually means more false positives.
bboxes = classifier.detectMultiScale(pixels,1.33,5)
# print bounding box for each detected face
for box in bboxes:
    # extract coordinates
    print(box)
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 3)

#scale picture according to scale_factor
scale_factor = 0.3
resize = cv.resize(pixels, None, fx = scale_factor, fy = scale_factor, interpolation=cv.INTER_LINEAR)
#show scaled picture
cv.imshow('output', resize)
# keep the window open until we press a key
cv.waitKey(0)
# close the window
cv.destroyAllWindows()