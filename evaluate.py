"""
General YOLO Model Evaluation Script

Usage:
    python evaluate.py
    Choose a dataset and enter the model path (weights/yolo11l.pt or similar)
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def find_datasets():
    """Find datasets in datasets/ directory."""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return []
    return [
        d.name
        for d in datasets_dir.iterdir()
        if d.is_dir() and (d / "data.yaml").exists()
    ]


def fix_data_yaml(data_yaml_path):
    """Fix data.yaml paths to match actual folder structure."""
    dataset_dir = data_yaml_path.parent

    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    folders = {}
    for split in ["train", "val", "test"]:
        if (dataset_dir / split / "images").exists():
            folders[split] = f"{split}/images"
        elif (dataset_dir / f"{split}ing" / "images").exists():
            folders[split] = f"{split}ing/images"
        elif split == "val" and (dataset_dir / "valid" / "images").exists():
            folders["val"] = "valid/images"

    modified = False
    for split, path in folders.items():
        if split in data and data[split] != path:
            print(f"  Fixing {split}: {data[split]} → {path}")
            data[split] = path
            modified = True

    if modified:
        with open(data_yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"✅ Fixed {data_yaml_path}")

    return data


def evaluate_model(model_path, dataset_name, split="val"):
    """Evaluate YOLO model on dataset."""
    data_yaml = Path(f"datasets/{dataset_name}/data.yaml")

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    if not data_yaml.exists():
        print(f"❌ Dataset not found: {data_yaml}")
        return

    print(f"🔧 Checking dataset paths...")
    fix_data_yaml(data_yaml)

    print(f"🚀 Evaluating {model_path} on {dataset_name}")

    try:
        model = YOLO(model_path)
        metrics = model.val(
            data=str(data_yaml.resolve()), split=split, save=False, verbose=False
        )

        print(f"\n📊 Results:")
        print(f"  mAP@0.5:      {metrics.box.map50:.3f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
        print(f"  Precision:    {metrics.box.mp:.3f}")
        print(f"  Recall:       {metrics.box.mr:.3f}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Check that your dataset has the right structure:")
        print("  datasets/your_dataset/")
        print("    ├── data.yaml")
        print("    ├── train/images/ & train/labels/")
        print("    └── val/images/ & val/labels/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model path (.pt file)")
    parser.add_argument("-d", "--dataset", help="Dataset name")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    if args.model and args.dataset:
        evaluate_model(args.model, args.dataset, args.split)
    else:
        datasets = find_datasets()
        if not datasets:
            print(
                "❌ No datasets found. Place datasets in 'datasets/' folder with data.yaml"
            )
            return

        print("Available datasets:")
        for i, d in enumerate(datasets, 1):
            print(f"  {i}. {d}")

        try:
            choice = int(input(f"Select dataset (1-{len(datasets)}): ")) - 1
            dataset = datasets[choice]
            model_path = input("Enter model path: ").strip()
            evaluate_model(model_path, dataset)
        except (ValueError, IndexError, KeyboardInterrupt):
            print("Invalid selection or interrupted")


if __name__ == "__main__":
    main()
