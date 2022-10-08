from pickle import TRUE
import numpy as np
import cv2 as cv
import PyCeres
import matplotlib.pyplot as plt

cap = cv.VideoCapture("./data/facevid_smallmotion.mp4")

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
image_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
image_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 20000,
                       qualityLevel = 1e-10,
                       minDistance = 5,
                       blockSize = 10 )

p0 = cv.goodFeaturesToTrack(old_gray, mask = None, useHarrisDetector= False, **feature_params)

inlier_mask = [True for i in p0]
print("before: "+ str(sum(inlier_mask)))

# cap.set(cv.CAP_PROP_POS_FRAMES,0) #reset to 1st frame
features_raw = []

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
    #save all features for each frame. apply boolean mask later to filter. Do this to keep track of each of the final features in each image
    features_raw.append(p1_fwd)
    
    #go through each feature and get the forward/backward status sf/sb and the tracked positions from optical flow backwards pbx pby
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
print(len(features))
#apply mask to the tracked features in each frame
features_i = []
for fr in features_raw:
    features_i.append(fr[inlier_mask].copy())
print(len(features_i[3]))
for f in features:
    cv.circle(old_frame,(int(f[0][0]),int(f[0][1])),3,(0,0,255),-1)

cv.imwrite('./data/face_features.png', old_frame)

cx = image_width/2
cy = image_height/2
poses = np.array([[0.0]*6]*count)
w_min = 0.01
w_max = 1.0
inv_depths = np.random.uniform(low = w_min, high = w_max, size = num_features)
f_init = image_width if image_width>image_height else image_height
k1_init = 0.0
k2_init = 0.0
variables = [1.0, k1_init*10.0, k2_init*10.0]

problem = PyCeres.Problem()
#go through each feature p0
for idx,f in enumerate(features):
    u0 = float(f[0][0])
    v0 = float(f[0][1])
    #for that feature p0, find the corresponding feature in each of the following frames
    for j,image in enumerate(features_i):
        u = image[idx][0][0]
        v = image[idx][0][1]
        cost_function = PyCeres.CreateBACostFunction(f_init, cx, cy,u0,v0,u,v)
        loss = PyCeres.HuberLoss(0.1)
        problem.AddResidualBlock(cost_function, loss, poses[j], inv_depths[idx], variables)
        #obtain camera poses (6 variables for each of the 30 frame) and inverse depths (1 variable for each of the 30 frames)

options = PyCeres.SolverOptions()
options.linear_solver_type = PyCeres.LinearSolverType.ITERATIVE_SCHUR
options.max_num_iterations = 100
options.minimizer_progress_to_stdout = True

summary = PyCeres.Summary()
PyCeres.Solve(options, problem, summary)

print(summary.BriefReport() + " \n")

print("poses")
print(poses)
print("inverse depths")
print(inv_depths)
print("variables")
print(variables)

cam_poses = []
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for camera in poses:
    rx = camera[0]
    ry = camera[1]
    rz = camera[2]
    tx = camera[3]
    ty = camera[4]
    tz = camera[5]

    x_cam = -tx-rz*ty+ry*tz
    y_cam = rz*tx-ty-rx*tz
    z_cam = -ry*tx+rx*ty-tz
    
    cam_poses.append([x_cam, y_cam, z_cam])

ax.scatter([i[0] for i in cam_poses],[i[1] for i in cam_poses],[i[2] for i in cam_poses])
plt.savefig('./datacam_poses_plt.png')
np.savetxt("./data/camera_poses.txt",np.array(cam_poses))

#Calibrate frames - no need, smartphone does it by itself -> should be OK. If not then fk me


