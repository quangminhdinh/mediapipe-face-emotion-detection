import time
from sklearn.metrics import accuracy_score
from .encode import extract_features
from .log import log_steps
from .model import fit_model
from .predict import predict

def preprocess_latency(paths):
    start = time.time()
    _ = [extract_features(path) for path in paths]
    total_latency = time.time() - start
    return total_latency / len(paths)

def preprocess_latency_log_steps(paths, split_path, label):
    total = len(paths)
    start = time.time()
    features = [log_steps(f"{split_path}/{path}", i, total, label) for i, path in enumerate(paths)]
    total_latency = time.time() - start
    latency = total_latency / total
    return features, latency

def model_fit_latency(model, data, labels, model_name, dataset_name, save=True):
    start = time.time()
    fit_model(model, data, labels, model_name, dataset_name, save)
    total_latency = time.time() - start
    return total_latency / len(labels)

def benchmark_prediction(model, test_data, labels):
    start = time.time()
    predicted = model.predict(test_data)
    acc = accuracy_score(labels, predicted)
    total_latency = time.time() - start
    latency = total_latency / len(labels)
    return acc, latency

def prediction_latency(model, dataset_name, image_path):
    start = time.time()
    result = predict(model, dataset_name, image_path)
    if not result:
        return
    latency = time.time() - start
    return result, latency
