import argparse
import csv
import numpy as np
from joblib import dump
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_dataset(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError("Dataset is empty")

    start_idx = 0
    try:
        [float(x) for x in rows[0]]
    except ValueError:
        start_idx = 1

    features = []
    labels = []
    for row in rows[start_idx:]:
        try:
            values = [float(x) for x in row]
        except ValueError:
            continue
        if len(values) < 2:
            continue
        features.append(values[:-1])
        labels.append(int(round(values[-1])))

    if not features:
        raise ValueError("No valid samples in dataset")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train a decision tree")
    parser.add_argument("--data", type=str, default="dataset.csv")
    parser.add_argument("--model", type=str, default="model.joblib")
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    args = parser.parse_args()

    X, y = load_dataset(args.data)

    if args.test_split and args.test_split > 0:
        stratify = y if len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_split,
            random_state=args.random_state,
            stratify=stratify,
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    clf = tree.DecisionTreeClassifier(
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    clf.fit(X_train, y_train)

    if X_test is not None:
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.3f} on {len(y_test)} samples")

    dump(clf, args.model)
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
