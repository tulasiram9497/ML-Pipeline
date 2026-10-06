from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ml_pipeline.data.loader import DataLoader
from ml_pipeline.data.validator import DataValidator
from ml_pipeline.tasks.detector import TaskDetector
from ml_pipeline.training.trainer import ModelTrainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoML Production Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Profile a dataset")
    profile_parser.add_argument("--path", required=True, help="Path to the dataset")

    train_parser = subparsers.add_parser("train", help="Train a supervised model")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--target", required=True)
    train_parser.add_argument("--task", default=None)
    train_parser.add_argument("--model", default="logistic_regression")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--target", required=True)
    evaluate_parser.add_argument("--model", default="logistic_regression")

    predict_parser = subparsers.add_parser("predict", help="Predict using a saved model bundle")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)

    return parser


def _profile_command(args):
    df = DataLoader.load_dataset(args.path)
    result = DataLoader.describe_dataset(df)
    print(json.dumps(result, indent=2, default=str))


def _train_command(args):
    df = DataLoader.load_dataset(args.data)
    validation = DataValidator.validate(df, target=args.target)
    if not validation.valid:
        raise ValueError("Dataset validation failed: " + "; ".join(validation.issues))
    task = TaskDetector.detect(df, args.target, override=args.task)
    result = ModelTrainer.train(df, target=args.target, task=task, model_name=args.model)
    print(json.dumps({"task": result.task, "metrics": result.metrics, "cv_summary": result.cv_summary}, indent=2, default=str))


def _evaluate_command(args):
    df = DataLoader.load_dataset(args.data)
    print({"status": "not-implemented", "data_shape": df.shape, "target": args.target})


def _predict_command(args):
    payload = json.loads(Path(args.input).read_text()) if Path(args.input).exists() else json.loads(args.input)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_columns = bundle["feature_columns"]
    frame = pd.DataFrame([payload])
    transformed = preprocessor.transform(frame[feature_columns])
    prediction = model.predict(transformed)
    print(json.dumps({"prediction": prediction.tolist()}))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    commands = {
        "profile": _profile_command,
        "train": _train_command,
        "evaluate": _evaluate_command,
        "predict": _predict_command,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
