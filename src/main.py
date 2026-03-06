"""Main orchestrator for experiment execution."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for experiment execution.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.method.type}")
    print("=" * 80)

    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("\nApplying sanity_check mode overrides...")

        # Override WandB project to avoid polluting main runs
        if cfg.wandb.project and not cfg.wandb.project.endswith("-sanity"):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"

        # Ensure online mode for WandB in sanity check
        cfg.wandb.mode = "online"

        print(f"  - WandB project: {cfg.wandb.project}")
        print(f"  - WandB mode: {cfg.wandb.mode}")
        print(f"  - Questions: {cfg.inference.sanity_num_samples}")

    elif cfg.mode == "main":
        print("\nRunning in main mode (full execution)...")
        cfg.wandb.mode = "online"
        print(f"  - WandB mode: {cfg.wandb.mode}")
        print(f"  - Questions: {cfg.dataset.num_questions}")

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # For inference-only tasks, run inference directly
    # (no separate training script needed)
    try:
        run_inference(cfg)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
