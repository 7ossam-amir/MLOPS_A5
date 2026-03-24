import os
import random
import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "assignment5")
    force_low = as_bool(os.getenv("FORCE_LOW_ACCURACY", "false"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
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
