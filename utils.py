import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(landmarks):
    l1, l2, l3 = landmarks
    v1 = l1 - l2
    v2 = l3 - l2
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def extract_features(landmarks, triplet_ids):
    triplet = landmarks[triplet_ids]
    return {
        "landmarks" : triplet,
        "angle" : angle_between(triplet)
    }

def extract_angles_only(landmarks, triplet_ids):
    triplet = landmarks[triplet_ids]
    return angle_between(triplet)

def get_mapper(domains):
    domain_to_label = {}
    for label, domain in enumerate(domains):
        domain_to_label[domain] = label
    return {
        "emotion_to_label": domain_to_label,
        "label_to_emotion": domains,
    }
