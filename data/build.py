import os

def load_path(split_path, domains):
    data_paths = {}
    for label in domains:
        if os.path.isdir(os.path.join(split_path, label)):
            data_paths[label] = os.listdir(f"{split_path}/{label}")
    return data_paths

def load_dataset(name):
    targetDir = f"{os.path.dirname(os.path.abspath(__file__))}/{name}"
    if not os.path.exists(targetDir):
        return
    config = __import__(f".{name}.config").DATA_CONFIG
    data_splits = {}
    for split in config["DATASETS"]:
        data_splits[split] = load_path(f"{config['DIR']}/{split}", config["DOMAINS"])
    return {
        "dataset": data_splits,
        "config": config,
    }
