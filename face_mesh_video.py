import cv2 as cv
from cv2 import flip
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

cap = cv.VideoCapture('./data/facevid_white_noglasses.mp4')
#configure face mesh
with mp_face_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, unflipped = cap.read()
        scale_factor = 0.3
        frame = cv.resize(cv.flip(unflipped,0), None, fx = scale_factor, fy = scale_factor, interpolation=cv.INTER_LINEAR)
        
        #if frame is not read successfully
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        #Set frame as non-writeable to improve performance
        frame.flags.writeable = False                
        #convert frame from BGR to RGB
        frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        #run face mesh process on grayscale frame
        results = face_mesh.process(frame)

        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5).process(frame)
        #set frame to writeable to draw face landmarks
        frame.flags.writeable = True
        #draw on frame according to face landmarks returned
        if face_detection.detections:
            for detection in face_detection.detections:
                mp_drawing.draw_detection(frame, detection)
        if results.multi_face_landmarks:
            #draw each face landmark
            for face_landmarks in results.multi_face_landmarks:
                #draw face tesselation
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                #draw face contours
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                #drawy eyes
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        #display frame with face mesh drawn, flip for selfie display
        cv.imshow('MediaPipe Face Mesh', cv.flip(frame, 1))
        
        #if user presses q, breaks out of while loop and closes window, stops video capture
        if cv.waitKey(1) == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
