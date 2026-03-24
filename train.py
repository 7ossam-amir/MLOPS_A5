import os
import random
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_training_data():
    data_path = os.getenv("DATA_PATH", "data/iris.csv")
    if os.path.exists(data_path):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        x_data = data[:, :-1]
        y_data = data[:, -1].astype(int)
        return x_data, y_data, f"csv:{data_path}"

    iris = load_iris()
    return iris.data, iris.target, "sklearn_builtin_iris"


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "assignment5")
    force_low = as_bool(os.getenv("FORCE_LOW_ACCURACY", "false"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    x_data, y_data, data_source = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    if force_low:
        # Deliberately scramble labels to make accuracy poor for failure demos.
        y_train = [random.choice([0, 1, 2]) for _ in y_train]

    model = LogisticRegression(max_iter=200)

    with mlflow.start_run() as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("force_low_accuracy", force_low)
        mlflow.log_param("data_source", data_source)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        with open("model_info.txt", "w", encoding="utf-8") as f:
            f.write(run_id)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
