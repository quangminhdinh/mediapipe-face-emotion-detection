import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import autosklearn.classification
from joblib import dump, load
from .config import KNN_CONFIG, SVM_CONFIG

def make_model(name="auto"):
    name = name.strip().lower()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=KNN_CONFIG["n_neighbors"])
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=SVM_CONFIG["C"], \
            gamma=SVM_CONFIG["gamma"], kernel=SVM_CONFIG["kernel"]))
    return autosklearn.classification.AutoSklearnClassifier()

def fit_model(model, data, labels, model_name, dataset_name, save=True):
    model.fit(data, labels)
    if save:
        outDir = f"{os.path.dirname(os.path.abspath(__file__))}/models/{dataset_name}"
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        save_dir = f"{outDir}/{model_name}.joblib"
        dump(model, save_dir)
        print(f"Model has been successfully saved at {save_dir}.")

def load_model(model_name, dataset_name):
    targetDir = f"{os.path.dirname(os.path.abspath(__file__))}/models/{dataset_name}"
    if not os.path.exists(targetDir):
        return
    targetPath = f"{targetDir}/{model_name}.joblib"
    if not os.path.exists(targetPath):
        return
    return load(targetPath)
