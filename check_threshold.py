import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def write_github_outputs(run_id: str, accuracy: float, threshold: float, passed: bool) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"accuracy={accuracy:.4f}\n")
        f.write(f"threshold={threshold:.2f}\n")
        f.write(f"passed={'true' if passed else 'false'}\n")


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
    model_info_path = os.getenv("MODEL_INFO_PATH", "model_info.txt")

    if not os.path.exists(model_info_path):
        print(f"ERROR: {model_info_path} not found.")
        return 1

    with open(model_info_path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: Run ID is empty.")
        return 1

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print(f"ERROR: accuracy metric not found for run {run_id}.")
        return 1

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {threshold:.2f}")
    passed = accuracy >= threshold
    write_github_outputs(run_id=run_id, accuracy=accuracy, threshold=threshold, passed=passed)

    if not passed:
        print("FAIL: Accuracy is below threshold.")
        return 1

    print("PASS: Accuracy meets threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
