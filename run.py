import sys
from argparse import ArgumentParser
from .preprocess import preprocess_loaded_all_splits, preprocess_data
from .model import make_model, load_model
from .benchmark import model_fit_latency, benchmark_prediction, prediction_latency

if __name__ == "__main__":
    parser = ArgumentParser(description="Build mediapipe face emotion detection models.")

    parser.add_argument("action", type=str, default="fit", choices=["fit", "predict", "benchmark"])
    parser.add_argument("--model", type=str, default="knn", choices=["auto", "knn", "svm"])
    parser.add_argument("--data", type=str, default="fer2013", choices=["fer2013"])
    parser.add_argument("--use-preprocessed-features", type=bool, default=True)

    parser.add_argument("--path", type=str)


    args = parser.parse_args()

    if args.action == "fit":
        if not args.model or not args.data:
            print("Missing arguments!")
            sys.exit()

        if args.use_preprocessed_features:
            features_output = preprocess_loaded_all_splits(args.data)
        else:
            features_output = preprocess_data(args.data)
        if not features_output:
            sys.exit()
        features, labels, _, mapper = features_output

        if "train" in labels:
            train_features = features["train"]
            train_labels = labels["train"]
        else:
            print("Training dataset not found, using the first subset found!")
            key = list(labels.keys())[0]
            train_features = features[key]
            train_labels = labels[key]

        model = make_model(args.model)
        fit_latency = model_fit_latency(model, train_features, train_labels, args.model, args.data)
        print(f"Average fit latency: {fit_latency}s")
        
        if "test" not in labels:
            print("Test dataset not found, cannot benchmark result model!")
            sys.exit()
        test_features = features["test"]
        test_labels = labels["test"]
        acc, predict_latency = benchmark_prediction(model, test_features, test_labels)
        print("Accuracy:", acc)
        print(f"Average prediction latency: {predict_latency}s")
    
    elif args.action == "predict":
        if not args.model or not args.data or not args.path:
            print("Missing arguments!")
            sys.exit()

        model = load_model(args.model, args.data)
        if not model:
            print("Model not found!")
            sys.exit()

        result, predict_latency = prediction_latency(model, args.data, args.path)
        if not result:
            sys.exit()
        print("Result:", result)
        print(f"Average prediction latency: {predict_latency}s")
    
    elif args.action == "benchmark":
        if not args.model or not args.data:
            print("Missing arguments!")
            sys.exit()

        if args.use_preprocessed_features:
            features_output = preprocess_loaded_all_splits(args.data)
        else:
            features_output = preprocess_data(args.data)
        if not features_output:
            sys.exit()
        features, labels, _, mapper = features_output

        model = load_model(args.model, args.data)
        if not model:
            print("Model not found!")
            sys.exit()
        
        if "test" not in labels:
            print("Test dataset not found, cannot benchmark model!")
            sys.exit()
        test_features = features["test"]
        test_labels = labels["test"]
        acc, predict_latency = benchmark_prediction(model, test_features, test_labels)
        print("Accuracy:", acc)
        print(f"Average prediction latency: {predict_latency}s")
