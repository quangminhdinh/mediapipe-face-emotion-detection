import cv2
import mediapipe as mp
import numpy as np
from .config import LANDMARK_IDS, ANGLE_IDS
from .utils import extract_angles_only

mp_face_mesh = mp.solutions.face_mesh

def get_face_mesh(image_file, label=None):
  with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    
    image = cv2.imread(image_file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      return
    normalized_landmarks = results.multi_face_landmarks[0].landmark
    face_landmarks = np.array([(normalized_landmarks[idx].x, normalized_landmarks[idx].y) for idx in LANDMARK_IDS])
    angles = np.array([extract_angles_only(face_landmarks, np.array(triplet)) for triplet in ANGLE_IDS])
    return {
        'landmarks' : face_landmarks,
        'angles' : angles,
        'label' : label
    }

def extract_features(image_file):
    return get_face_mesh(image_file)['angles']
