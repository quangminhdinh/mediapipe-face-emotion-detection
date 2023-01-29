import os
from .utils import get_mapper
from .encode import extract_features

def predict(model, dataset_name, image_path):
    targetDir = f"{os.path.dirname(os.path.abspath(__file__))}/data/{dataset_name}"
    if not os.path.exists(f"{os.path.dirname(os.path.abspath(__file__))}/{image_path}") \
        or not os.path.exists(image_path):
        print("Invalid path!")
        return
    if not os.path.exists(targetDir):
        print("Dataset configurations not found!")
        return
    config = __import__(f".data.{dataset_name}.config").DATA_CONFIG
    label_to_emotion = get_mapper(config["DOMAINS"])["label_to_emotion"]
    features = extract_features(image_path)
    predicted = model.predict(features)
    return label_to_emotion[predicted]
