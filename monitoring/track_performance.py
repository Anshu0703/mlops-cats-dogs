"""
Simulates a batch of real-world requests against the live API using
held-out test images (known true labels), then reports accuracy and
other performance metrics — the kind of check you'd run periodically
in production to catch model drift.

Usage:
    python monitoring/track_performance.py --host http://localhost:8000 --sample-size 100
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import requests

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "test"
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def collect_sample(sample_size: int, seed: int = 42):
    """Gathers a random sample of (filepath, true_label) pairs from the test set."""
    samples = []
    for label in ["cat", "dog"]:
        class_dir = TEST_DIR / label
        for f in class_dir.glob("*.jpg"):
            samples.append((f, label))

    random.Random(seed).shuffle(samples)
    return samples[:sample_size]


def run_batch(host: str, samples: list):
    """Sends each sample image to /predict and compares against the true label."""
    results = []

    for filepath, true_label in samples:
        with open(filepath, "rb") as f:
            response = requests.post(
                f"{host}/predict",
                files={"file": (filepath.name, f, "image/jpeg")},
                timeout=10,
            )
        response.raise_for_status()
        prediction = response.json()

        results.append({
            "filename": filepath.name,
            "true_label": true_label,
            "predicted_label": prediction["predicted_class"],
            "confidence": prediction["confidence"],
            "correct": true_label == prediction["predicted_class"],
        })

    return results


def summarize(results: list):
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total else 0.0

    per_class = {}
    for label in ["cat", "dog"]:
        class_results = [r for r in results if r["true_label"] == label]
        class_correct = sum(r["correct"] for r in class_results)
        per_class[label] = {
            "count": len(class_results),
            "correct": class_correct,
            "accuracy": class_correct / len(class_results) if class_results else 0.0,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": total,
        "overall_accuracy": round(accuracy, 4),
        "per_class": per_class,
    }


def main(host: str, sample_size: int):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {sample_size} samples from test set...")
    samples = collect_sample(sample_size)

    print(f"Sending {len(samples)} requests to {host}/predict ...")
    results = run_batch(host, samples)

    summary = summarize(results)
    summary["results"] = results

    report_path = REPORT_DIR / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall accuracy: {summary['overall_accuracy']:.2%}")
    for label, stats in summary["per_class"].items():
        print(f"  {label}: {stats['correct']}/{stats['count']} ({stats['accuracy']:.2%})")
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://localhost:8000")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    main(host=args.host, sample_size=args.sample_size)