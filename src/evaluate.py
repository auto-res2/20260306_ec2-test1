"""Evaluation script for comparing runs using WandB API."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help="JSON string list of run IDs to compare",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (optional, can use env var WANDB_ENTITY)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project (optional, can use env var WANDB_PROJECT)",
    )

    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()

    # Find runs by display name (most recent first)
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(f"No runs found with display name: {run_id}")

    run = runs[0]  # Get most recent run with this name

    print(f"Fetched run: {run_id} (WandB ID: {run.id})")

    return {
        "run_id": run_id,
        "wandb_id": run.id,
        "config": dict(run.config),
        "summary": dict(run.summary),
        "history": run.history(),
    }


def export_per_run_metrics(run_data: Dict[str, Any], output_dir: Path) -> None:
    """Export per-run metrics and figures.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export summary metrics
    metrics = {
        "run_id": run_data["run_id"],
        "accuracy": run_data["summary"].get("accuracy", None),
        "num_questions": run_data["summary"].get("num_questions", None),
        "avg_verifiability": run_data["summary"].get("avg_verifiability", None),
        "avg_steps": run_data["summary"].get("avg_steps", None),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Exported metrics to {output_dir / 'metrics.json'}")

    # Create per-run figures if history exists
    history = run_data["history"]
    if not history.empty and "_step" in history.columns:
        # Figure: accuracy over steps (if logged)
        if "accuracy" in history.columns:
            plt.figure(figsize=(8, 6))
            plt.plot(history["_step"], history["accuracy"])
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy - {run_data['run_id']}")
            plt.grid(True)
            plt.tight_layout()
            output_path = output_dir / f"accuracy_{run_data['run_id']}.pdf"
            plt.savefig(output_path)
            plt.close()
            print(f"  Saved figure: {output_path}")


def generate_comparison_plots(
    all_run_data: List[Dict[str, Any]], output_dir: Path
) -> None:
    """Generate comparison plots across runs.

    Args:
        all_run_data: List of run data dictionaries
        output_dir: Output directory for comparison plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Extract metrics for comparison
    run_ids = [r["run_id"] for r in all_run_data]
    accuracies = [r["summary"].get("accuracy", 0.0) for r in all_run_data]

    # Figure 1: Bar chart of accuracy
    plt.figure(figsize=(10, 6))
    colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
    plt.bar(run_ids, accuracies, color=colors)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.ylim([0, 1.0])
    plt.tight_layout()
    output_path = output_dir / "comparison_accuracy.pdf"
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved comparison figure: {output_path}")

    # Figure 2: Verifiability comparison (if available)
    verifiabilities = [r["summary"].get("avg_verifiability") for r in all_run_data]
    if any(v is not None for v in verifiabilities):
        plt.figure(figsize=(10, 6))
        valid_runs = [
            (rid, v) for rid, v in zip(run_ids, verifiabilities) if v is not None
        ]
        if valid_runs:
            rids, vals = zip(*valid_runs)
            plt.bar(rids, vals, color="#9b59b6")
            plt.xlabel("Method")
            plt.ylabel("Avg Verifiability Score")
            plt.title("Verifiability Comparison")
            plt.xticks(rotation=45, ha="right")
            plt.ylim([0, 1.0])
            plt.tight_layout()
            output_path = output_dir / "comparison_verifiability.pdf"
            plt.savefig(output_path)
            plt.close()
            print(f"  Saved comparison figure: {output_path}")

    # Figure 3: Steps comparison (if available)
    steps = [r["summary"].get("avg_steps") for r in all_run_data]
    if any(s is not None for s in steps):
        plt.figure(figsize=(10, 6))
        valid_runs = [(rid, s) for rid, s in zip(run_ids, steps) if s is not None]
        if valid_runs:
            rids, vals = zip(*valid_runs)
            plt.bar(rids, vals, color="#e74c3c")
            plt.xlabel("Method")
            plt.ylabel("Avg Number of Steps")
            plt.title("Average Steps Comparison")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            output_path = output_dir / "comparison_steps.pdf"
            plt.savefig(output_path)
            plt.close()
            print(f"  Saved comparison figure: {output_path}")


def export_aggregated_metrics(
    all_run_data: List[Dict[str, Any]], output_path: Path
) -> None:
    """Export aggregated metrics across runs.

    Args:
        all_run_data: List of run data dictionaries
        output_path: Output file path
    """
    # Collect metrics by run_id
    metrics_by_run = {}
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        metrics_by_run[run_id] = {
            "accuracy": run_data["summary"].get("accuracy", None),
            "num_questions": run_data["summary"].get("num_questions", None),
            "avg_verifiability": run_data["summary"].get("avg_verifiability", None),
            "avg_steps": run_data["summary"].get("avg_steps", None),
        }

    # Identify best proposed and best baseline
    proposed_runs = {k: v for k, v in metrics_by_run.items() if "proposed" in k}
    baseline_runs = {k: v for k, v in metrics_by_run.items() if "comparative" in k}

    best_proposed = None
    best_proposed_acc = -1
    for run_id, metrics in proposed_runs.items():
        if metrics["accuracy"] is not None and metrics["accuracy"] > best_proposed_acc:
            best_proposed = run_id
            best_proposed_acc = metrics["accuracy"]

    best_baseline = None
    best_baseline_acc = -1
    for run_id, metrics in baseline_runs.items():
        if metrics["accuracy"] is not None and metrics["accuracy"] > best_baseline_acc:
            best_baseline = run_id
            best_baseline_acc = metrics["accuracy"]

    # Compute gap
    gap = None
    if best_proposed_acc >= 0 and best_baseline_acc >= 0:
        gap = best_proposed_acc - best_baseline_acc

    # Assemble aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_proposed_accuracy": best_proposed_acc if best_proposed_acc >= 0 else None,
        "best_baseline": best_baseline,
        "best_baseline_accuracy": best_baseline_acc if best_baseline_acc >= 0 else None,
        "gap": gap,
    }

    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated metrics saved to {output_path}")
    print(f"  Best proposed: {best_proposed} (accuracy: {best_proposed_acc:.4f})")
    print(f"  Best baseline: {best_baseline} (accuracy: {best_baseline_acc:.4f})")
    if gap is not None:
        print(f"  Gap: {gap:+.4f}")


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Parse run_ids
    run_ids = json.loads(args.run_ids)

    # Get WandB credentials
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    project = args.wandb_project or os.environ.get("WANDB_PROJECT")

    if not entity or not project:
        raise ValueError(
            "WandB entity and project must be provided via args or environment variables"
        )

    print(f"Fetching runs from WandB: {entity}/{project}")
    print(f"Run IDs: {run_ids}\n")

    # Fetch run data
    all_run_data = []
    for run_id in run_ids:
        try:
            run_data = fetch_run_data(entity, project, run_id)
            all_run_data.append(run_data)
        except Exception as e:
            print(f"Warning: Failed to fetch run {run_id}: {e}")
            continue

    if not all_run_data:
        print("Error: No valid runs fetched")
        return

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Export per-run metrics and figures
    print("\nExporting per-run metrics...")
    for run_data in all_run_data:
        output_dir = results_dir / run_data["run_id"]
        export_per_run_metrics(run_data, output_dir)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    comparison_dir = results_dir / "comparison"
    generate_comparison_plots(all_run_data, comparison_dir)

    # Export aggregated metrics
    print("\nExporting aggregated metrics...")
    export_aggregated_metrics(all_run_data, comparison_dir / "aggregated_metrics.json")

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
