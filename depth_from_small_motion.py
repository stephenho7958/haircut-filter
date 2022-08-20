from pickle import TRUE
import numpy as np
import cv2 as cv
import mediapipe as mp


cap = cv.VideoCapture('./data/facevid_smallmotion.mp4')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 20000,
                       qualityLevel = 1e-10,
                       minDistance = 5,
                       blockSize = 10 )

p0 = cv.goodFeaturesToTrack(old_gray, mask = None, useHarrisDetector= False, **feature_params)

inlier_mask = [True for i in p0]
print("before: "+ str(sum(inlier_mask)))

count=0
while(1):
    count+=1
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    p1_fwd, st_fwd, err_fwd = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    p0_bwd, st_bwd, err_bwd = cv.calcOpticalFlowPyrLK(frame_gray, old_gray, p1_fwd, None, **lk_params)
    
    for idx, i in enumerate(p0):
        sf = st_fwd[idx]
        sb = st_bwd[idx]
        p0x = i[0][0]
        pbx =  p0_bwd[idx][0][0]
        p0y = i[0][1]
        pby =  p0_bwd[idx][0][1]
        #reject features based on status and bidirectional error
        if(sf == 0 or sb == 0  or np.sqrt((p0x-pbx)**2 + (p0y-pby)**2)>0.1):
            inlier_mask[idx] = False
    if count>=30:
            break
num_features = sum(inlier_mask)
print("after: "+str(num_features))

#apply mask to original features
features = p0[inlier_mask].copy()
for f in features:
    cv.circle(old_frame,(int(f[0][0]),int(f[0][1])),3,(0,0,255),-1)

cv.imwrite('./data/face_features.png', old_frame)