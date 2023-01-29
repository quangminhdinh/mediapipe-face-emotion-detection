# Face emotion detection using MediaPipe and angular encoding

This repository is a replication of the techniques introduced in [Deploying Machine Learning Techniques for Human Emotion Detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8828335/).

The replication was implemented using the [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz) dataset, without the implementation of a super-resolution model in the data preprocessing stage. Encoded features are processed using a Support Vector Machine (SVM), k-Nearest Neighbor (KNN) classifier, or [autosklearn](https://automl.github.io/auto-sklearn/master/).

```bibtex
@article{10.1155/2022/8032673,
author = {Siam, Ali I. and Soliman, Naglaa F. and Algarni, Abeer D. and Abd El-Samie, Fathi E. and Sedik, Ahmed and Ding, Bai Yuan},
title = {Deploying Machine Learning Techniques for Human Emotion Detection},
year = {2022},
issue_date = {2022},
publisher = {Hindawi Limited},
address = {London, GBR},
volume = {2022},
issn = {1687-5265},
url = {https://doi.org/10.1155/2022/8032673},
doi = {10.1155/2022/8032673},
journal = {Intell. Neuroscience}
}
```

## Setup

Both the SVM and KNN classifiers have the same hyperparameters as introduced in the paper, and can be found in `config.py`.

The dataset used by this replication is [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz) dataset. You have to download it and put the training and testing subset in `data/fer2013/train/` and `data/fer2013/test/` accordingly.

## Requirements

To install requirements:

```setup
pip install scikit-learn
pip install mediapipe
pip install opencv-python
pip install joblib
```

If you want to use autokeras:

```setup
pip install autokeras
```

## Training

To fit a model, run this command:

```train
python run.py fit --model [MODEL_NAME] --data [DATASET_NAME] [--use-preprocessed-features]
```

- `[MODEL_NAME]` can be `knn`, `svm`, or `auto`, default is `knn`. All exported model can be found in `models/[DATASET_NAME]/[MODEL_NAME].joblib`. Saved autosklearn model can be downloaded [here](https://drive.google.com/file/d/1bZTUxfSiOE0ReaQDBHkKm1WfhetKXGPy/view?usp=sharing).
- `[DATASET_NAME]`: For now, only `fer2013` is supported, default is `fer2013`. All datasets will be saved in `data`.
- `[--use-preprocessed-features]`: Add if you want to use encoded features saved in `features/[DATASET_NAME]/[SUBSET]/[LABEL].csv` instead of processing the dataset from the beginning.

## Evaluation

To evaluate a model, run:

```eval
python run.py benchmark --model [MODEL_NAME] --data [DATASET_NAME] [--use-preprocessed-features]
```

- `[MODEL_NAME]`, `[DATASET_NAME]` and `[--use-preprocessed-features]` are the same as the training command.

## Predict

To use a model to predict an image, run:

```predict
python run.py predict --model [MODEL_NAME] --data [DATASET_NAME] --path [IMAGE_PATH] [--use-preprocessed-features]
```

- `[MODEL_NAME]`, `[DATASET_NAME]` and `[--use-preprocessed-features]` are the same as the training command.
- `[IMAGE_PATH]` is the path to the image that you want to predict.

## Results

Since the [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz) dataset contains only 48x48 images, and no super-resolution model was implemented, the accuracy of each model is inferior compared to the result produced by the paper (knn = 0.48, svm = 0.40, autokeras = 0.52).
