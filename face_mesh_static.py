import cv2 as cv
import mediapipe as mp
def plot_coordinates(landmarks):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    landmark_x, landmark_y, landmark_z = ([] for i in range(3))

    for landmark in landmarks.landmark:
        landmark_x.append(landmark.x)
        landmark_y.append(landmark.y)
        landmark_z.append(-1*landmark.z)
    
    ax.cla()
    ax.set_xlim3d(0.2, 0.8)
    ax.set_ylim3d(0.2,0.8)
    ax.set_zlim3d(-1, 1)
    ax.scatter(landmark_x,landmark_y,landmark_z, marker='.')
    plt.show()

    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# List of image files
IMAGE_FILES = ["./test_images/test5.JPG"]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#configure face mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    #loop through each image file in IMAGE_FILES list
    for idx, file in enumerate(IMAGE_FILES):
        #read image
        image = cv.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        #If no landmarks, continue to next image
        if not results.multi_face_landmarks:
            continue
        #create copy of image file read
        annotated_image = image.copy()
        #draw landmarks on face
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            print(idx)
            
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

print("This is a test pirint:")
print(face_landmarks.landmark[0].x)
plot_coordinates(face_landmarks)
#scale picture according to scale_factor
# scale_factor = 0.7
# resize = cv.resize(annotated_image, None, fx = scale_factor, fy = scale_factor, interpolation=cv.INTER_LINEAR)
# #show scaled picture
# cv.imshow('output', resize)
# # keep the window open until we press a key
# cv.waitKey(0)
# # close the window
# cv.destroyAllWindows()
