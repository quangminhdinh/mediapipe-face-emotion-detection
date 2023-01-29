import os

DATA_CONFIG = {
    "NAME": "fer2013",
    "DIR": os.path.dirname(os.path.abspath(__file__)),
    "DATASETS": ['train', 'test'],
    "DOMAINS": (
        "neutral",
        "angry",
        "fear",
        "disgust",
        "surprise",
        "sad",
        "happy",
    ),
}
