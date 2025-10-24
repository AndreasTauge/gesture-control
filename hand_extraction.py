import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
import cv2 as cv
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on image"""
    annotated_image = np.copy(rgb_image)
    
    # loop through detected hands
    for hand_landmarks in detection_result.hand_landmarks:
        # convert to proto format
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])
        
        # draw landmarks
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
    
    return annotated_image

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(base_options=base_options, num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

test_image = mp.Image.create_from_file("data/thumbs_down/img_0.jpg")

detection_result = detector.detect(test_image)

annotated_image = draw_landmarks_on_image(test_image.numpy_view(), detection_result)

cv.imshow("Hand Detection", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
cv.waitKey(0)  
cv.destroyAllWindows()