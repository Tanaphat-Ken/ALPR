"""Evaluate all checkpoints in an experiment directory.

This script loads each checkpoint and runs evaluation on the test set,
creating eval_test.json files for each checkpoint.

Usage:
    python train/eval_checkpoints.py --experiment-dir outputs/grid/exp1_v1_clean
    python train/eval_checkpoints.py --experiment-dir outputs/grid/exp1_v1_clean --split test --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import List, Optional

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("eval_checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in an experiment")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Experiment directory containing checkpoints")
    parser.add_argument("--csv", type=str, help="Dataset CSV (if not specified, will look for training_args)")
    parser.add_argument("--data-root", type=str, action="append", help="Data root directories")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--model-id", type=str, help="Base model ID (if not specified, will detect from config)")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing eval results")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations in parallel (experimental)")
    return parser.parse_args()


def find_checkpoints(experiment_dir: Path) -> List[Path]:
    """Find all checkpoint directories."""
    checkpoints = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            # Extract checkpoint number for sorting
            match = re.search(r"checkpoint-(\d+)", item.name)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, item))
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: x[0])
    return [path for _, path in checkpoints]


def load_training_config(experiment_dir: Path) -> dict:
    """Load training configuration from the experiment directory."""
    config = {}
    
    # Try to load training_args.bin (contains original training arguments)
    training_args_file = experiment_dir / "training_args.bin"
    if training_args_file.exists():
        try:
            import torch
            training_args = torch.load(training_args_file, map_location="cpu")
            if hasattr(training_args, '__dict__'):
                config.update(training_args.__dict__)
            logger.info("Loaded training arguments from training_args.bin")
        except Exception as e:
            logger.warning(f"Could not load training_args.bin: {e}")
    
    # Load config.json for model info
    config_file = experiment_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            config['model_config'] = model_config
    
    return config


def build_eval_command(checkpoint_dir: Path, args: argparse.Namespace, config: dict) -> List[str]:
    """Build evaluation command for a checkpoint."""
    eval_script = SCRIPT_DIR / "eval_trocr.py"
    
    cmd = [
        sys.executable, str(eval_script),
        "--model-path", str(checkpoint_dir),
        "--split", args.split,
        "--num-beams", str(args.num_beams),
        "--batch-size", str(args.batch_size),
    ]
    
    # CSV and data roots
    if args.csv:
        cmd.extend(["--csv", args.csv])
    elif 'train_dataset' in config:
        # Try to extract from training config
        logger.warning("No CSV specified, hoping eval_trocr.py can infer from checkpoint")
    
    if args.data_root:
        for root in args.data_root:
            cmd.extend(["--data-root", root])
    
    # Model ID
    if args.model_id:
        cmd.extend(["--model-id", args.model_id])
    elif 'model_config' in config and '_name_or_path' in config['model_config']:
        model_id = config['model_config']['_name_or_path']
        cmd.extend(["--model-id", model_id])
        logger.info(f"Using model ID from config: {model_id}")
    
    # Province prediction (detect from config)
    if 'predict_province' in config and config['predict_province']:
        cmd.append("--predict-province")
        if 'province_format' in config:
            cmd.extend(["--province-format", config['province_format']])
    
    # Other options
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    
    # LoRA detection
    lora_files = [
        checkpoint_dir / "adapter_model.safetensors",
        checkpoint_dir / "adapter_model.bin",
        checkpoint_dir / "adapter_model.pt"
    ]
    if any(f.exists() for f in lora_files):
        cmd.append("--use-lora")
        logger.info(f"Detected LoRA adapters in {checkpoint_dir.name}")
    
    # Save results
    results_file = checkpoint_dir / f"eval_{args.split}.json"
    cmd.extend(["--save-results", str(results_file)])
    
    return cmd


def run_evaluation(checkpoint_dir: Path, args: argparse.Namespace, config: dict) -> bool:
    """Run evaluation for a single checkpoint."""
    results_file = checkpoint_dir / f"eval_{args.split}.json"
    
    # Skip if results exist and not overwriting
    if results_file.exists() and not args.overwrite:
        logger.info(f"Skipping {checkpoint_dir.name} - results already exist")
        return True
    
    logger.info(f"Evaluating checkpoint: {checkpoint_dir.name}")
    
    # Build and run command
    cmd = build_eval_command(checkpoint_dir, args, config)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ Successfully evaluated {checkpoint_dir.name}")
            return True
        else:
            logger.error(f"✗ Failed to evaluate {checkpoint_dir.name}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Exception evaluating {checkpoint_dir.name}: {e}")
        return False


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory does not exist: {experiment_dir}")
        return
    
    # Find checkpoints
    checkpoints = find_checkpoints(experiment_dir)
    if not checkpoints:
        logger.error(f"No checkpoints found in {experiment_dir}")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Load training configuration
    config = load_training_config(experiment_dir)
    
    # Evaluate each checkpoint
    successful = 0
    failed = 0
    
    for checkpoint_dir in checkpoints:
        if run_evaluation(checkpoint_dir, args, config):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Evaluation complete: {successful} successful, {failed} failed")
    
    # Create summary
    summary = {
        "experiment": experiment_dir.name,
        "total_checkpoints": len(checkpoints),
        "successful_evaluations": successful,
        "failed_evaluations": failed,
        "checkpoints": []
    }
    
    for checkpoint_dir in checkpoints:
        results_file = checkpoint_dir / f"eval_{args.split}.json"
        checkpoint_info = {
            "checkpoint": checkpoint_dir.name,
            "step": int(re.search(r"checkpoint-(\d+)", checkpoint_dir.name).group(1)),
            "eval_file": str(results_file.relative_to(experiment_dir)),
            "exists": results_file.exists()
        }
        
        # Load metrics if available
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    checkpoint_info["metrics"] = eval_data.get("metrics", {})
            except Exception as e:
                logger.warning(f"Could not load metrics from {results_file}: {e}")
        
        summary["checkpoints"].append(checkpoint_info)
    
    # Save summary
    summary_file = experiment_dir / f"checkpoint_evaluations_{args.split}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved evaluation summary to {summary_file}")


if __name__ == "__main__":
    main()