import os
import pandas as pd
from sklearn.utils import shuffle   
from .data.build import load_dataset
from .benchmark import preprocess_latency_log_steps
from .utils import get_mapper

def preprocess_split(split, split_data, domains, dataset_path, dataset_name, export_features=True):
    features = {}

    for label in domains:
        print(f"Start preprocessing {split}:")
        raw_features, latency = preprocess_latency_log_steps(split_data[label], \
            f"{dataset_path}/{split}", label)
        print(f"Average preprocess latency: {latency}s")
        features[label] = [feature for feature in raw_features if feature is not None]
        
        if export_features:
            outDir = f"{os.path.dirname(os.path.abspath(__file__))}/features/{dataset_name}"
            if not os.path.exists(outDir):
                os.mkdir(outDir)
            outSplitDir = f"{outDir}/{split}"
            if not os.path.exists(outSplitDir):
                os.mkdir(outSplitDir)
            
            features_df = pd.DataFrame(features[label])
            export_dir = f"{outSplitDir}/{label}.csv"
            features_df.to_csv(export_dir, index=False)
            print(f"Features have been successfully exported at {export_dir}.")
    return features

def prepare_split_label(split_features, domains):
    labels = []
    features = []

    for label, domain in enumerate(domains):
        labels += len(split_features[domain]) * [label]
        features += split_features[domain]
    return features, labels

def preprocess_data(dataset_name, shuffle_data=True):
    dataset = load_dataset(dataset_name)
    if not dataset:
        print("Dataset not found!")
        return
    data_splits = dataset["dataset"]
    config = dataset["config"]
    
    raw_features = {}
    features = {}
    labels = {}

    for split in config["DATASETS"]:
        raw_features[split] = preprocess_split(split, data_splits[split], config["DOMAINS"], \
            config["DIR"], config["NAME"])
        split_features, split_labels = prepare_split_label(raw_features[split], config["DOMAINS"])
        if shuffle_data:
            split_features, split_labels = shuffle(split_features, split_labels)
        features[split] = split_features
        labels[split] = split_labels
        
    return features, labels, raw_features, get_mapper(config["DOMAINS"])

def load_features(path):
    features = {}
    for label in os.listdir(path):
        features_df = pd.read_csv(f"{path}/{label}")
        features[label[:-4]] = features_df.to_numpy()
    return features

def preprocess_loaded_features(raw_features, dataset_name, shuffle_data=True):
    targetDir = f"{os.path.dirname(os.path.abspath(__file__))}/data/{dataset_name}"
    domains = list(raw_features.keys())
    if os.path.exists(targetDir):
        domains = __import__(f".data.{dataset_name}.config").DATA_CONFIG["DOMAINS"]
    else:
        print("Dataset configurations not found, fallback to using system's files order.")
    features, labels = prepare_split_label(raw_features, domains)
    if shuffle_data:
        features, labels = shuffle(features, labels)
    return features, labels, domains

def preprocess_loaded_all_splits(dataset_name, shuffle_data=True):
    targetDir = f"{os.path.dirname(os.path.abspath(__file__))}/features/{dataset_name}"
    if not os.path.exists(targetDir):
        print("Dataset's features not found!")
        return
    if len(os.listdir(targetDir) == 0):
        print("Features not found!")
        return
        
    raw_features = {}
    features = {}
    labels = {}

    for split_name in os.listdir(targetDir):
        targetSplit = f"{targetDir}/{split_name}"
        raw_features[split_name] = load_features(targetSplit)
        split_features, split_labels, domains = preprocess_loaded_features(raw_features[split_name], \
            dataset_name, shuffle_data)
        features[split_name] = split_features
        labels[split_name] = split_labels
    return features, labels, raw_features, get_mapper(domains)
