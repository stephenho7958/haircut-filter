from pickle import TRUE
import numpy as np
import cv2 as cv
import mediapipe as mp

def createMask(ymin:int, xmin:int, width:int, height:int, frame):
    # print(str(ymin)+", "+str(xmin)+", "+str(width)+", "+str(height))
    mask =  np.zeros_like(frame)
    mask[ymin:ymin+height,xmin:xmin+width,:] = frame[ymin:ymin+height,xmin:xmin+width,:]
    return mask



#face detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection



cap = cv.VideoCapture('./data/facevid_smallmotion.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 20000,
                       qualityLevel = 1e-10,
                       minDistance = 7,
                       blockSize = 10 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
# while(True):
#     ret, old_frame = cap.read()

#     face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5).process(old_frame)
#     if face_detection.detections:
#         face_detection.detections[0].location_data.relative_bounding_box.xmin-=(face_detection.detections[0].location_data.relative_bounding_box.width/4)
#         face_detection.detections[0].location_data.relative_bounding_box.ymin-=(face_detection.detections[0].location_data.relative_bounding_box.height/4)
#         face_detection.detections[0].location_data.relative_bounding_box.width*=1.5
#         face_detection.detections[0].location_data.relative_bounding_box.height*=1.5
#         face_mask = createMask(ymin=int(face_detection.detections[0].location_data.relative_bounding_box.ymin*1920),
#         xmin=int(face_detection.detections[0].location_data.relative_bounding_box.xmin*1080),
#         width=int(face_detection.detections[0].location_data.relative_bounding_box.width*1080),
#         height=int(face_detection.detections[0].location_data.relative_bounding_box.height*1920), frame = old_frame)
#         break
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# face_mask_gray = cv.cvtColor(face_mask, cv.COLOR_BGR2GRAY)

# p0 = cv.cornerHarris(np.float32(old_gray),2,3,0.04)

p0 = cv.goodFeaturesToTrack(old_gray, mask = None, useHarrisDetector= False, **feature_params)
# print(p0[1][0][0])
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
count=0
while(1):
    count+=1
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5).process(frame)
    # if face_detection.detections:
    #     face_detection.detections[0].location_data.relative_bounding_box.xmin-=(face_detection.detections[0].location_data.relative_bounding_box.width/4)
    #     face_detection.detections[0].location_data.relative_bounding_box.ymin-=(face_detection.detections[0].location_data.relative_bounding_box.height/4)
    #     face_detection.detections[0].location_data.relative_bounding_box.width*=1.5
    #     face_detection.detections[0].location_data.relative_bounding_box.height*=1.5
    #     face_mask = createMask(ymin=int(face_detection.detections[0].location_data.relative_bounding_box.ymin*1920),
    #     xmin=int(face_detection.detections[0].location_data.relative_bounding_box.xmin*1080),
    #     width=int(face_detection.detections[0].location_data.relative_bounding_box.width*1080),
    #     height=int(face_detection.detections[0].location_data.relative_bounding_box.height*1920), frame = frame)
    # face_mask_gray = cv.cvtColor(face_mask, cv.COLOR_BGR2GRAY)

        

    # calculate optical flow

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if(count==3):
        print(err)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # print(str(int(a))+", "+str(int(b))+", "+str(int(c))+", "+str(int(d)))
        try:
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        except:
            continue
    img = cv.add(frame, mask)
    scale_factor = 0.3
    resize = cv.resize(img, None, fx = scale_factor, fy = scale_factor, interpolation=cv.INTER_LINEAR)
    cv.imshow('frame', cv.flip(resize,0))
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    #update features if every 10 loops
    if(count%10==0):
        np.concatenate([p0,cv.goodFeaturesToTrack(old_gray, mask = None, useHarrisDetector= True, **feature_params)])
        # p0 = cv.goodFeaturesToTrack(old_gray, mask = None, useHarrisDetector= True, **feature_params)
    #     print("----------------------------------")
    #     print(len(p0))
        



    if cv.waitKey(1) == ord('q'):
            break
cv.destroyAllWindows()