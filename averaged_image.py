import numpy as np
import cv2 as cv

image_data = []
cap = cv.VideoCapture('./data/facevid_smallmotion.mp4')

while True:
    ret, frame = cap.read()
    #check frame read correctly
    if not ret:
        print("cannot receive frame. Exiting ...")
        break
    image_data.append(frame)

avg_image = image_data[0]
for i in range(len(image_data)):
    if i == 0:
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        avg_image = cv.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

cv.imwrite('./data/avg_face.png', avg_image)