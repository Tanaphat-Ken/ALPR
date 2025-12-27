"""Automation helper to execute image-based province prediction experiments.

This script runs multi-task TrOCR experiments where province prediction is done
via direct image classification rather than text parsing. Each experiment
corresponds to the original text-based experiments but uses the multi-task model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_multitask_trocr.py"
# TODO: Create eval_multitask_trocr.py for image-based evaluation
EVAL_SCRIPT = SCRIPT_DIR / "eval_multitask_trocr.py"


@dataclass(frozen=True)
class ImageBasedExperiment:
    key: str
    description: str
    base_model: str
    model_path: str | None
    augment: str
    use_fp16: bool = False
    output_name: str | None = None

    def output_dir(self, base_dir: Path) -> Path:
        return base_dir / (self.output_name or self.key)

    def train_args(self) -> List[str]:
        """Generate training arguments for multi-task TrOCR."""
        args = [
            "--model-id", self.base_model,
            "--augment", self.augment,
        ]
        
        if self.model_path:
            args.extend(["--model-path", self.model_path])
            
        if self.use_fp16:
            args.append("--fp16")
            
        return args

    def eval_args(self) -> List[str]:
        """Generate evaluation arguments for multi-task TrOCR."""
        args = [
            "--model-id", self.base_model,
            "--split", "test",
        ]
        return args


# Image-based experiments (exp6-exp10) - mirrors of exp1-exp5 but with image-based province prediction
IMAGE_EXPERIMENTS: List[ImageBasedExperiment] = [
    ImageBasedExperiment(
        key="exp6_v1_clean_image",
        description="V1 (.pth) baseline + clean data + IMAGE-based province prediction",
        base_model="openthaigpt/thai-trocr",
        model_path="models/weights/charactor_reader.pth",
        augment="none",
        use_fp16=False,
        output_name="exp6_v1_clean_image",
    ),
    ImageBasedExperiment(
        key="exp7_v1_aug_image", 
        description="V1 (.pth) + heavy augmentation + IMAGE-based province prediction",
        base_model="openthaigpt/thai-trocr",
        model_path="models/weights/charactor_reader.pth",
        augment="heavy",
        use_fp16=True,
        output_name="exp7_v1_aug_image",
    ),
    ImageBasedExperiment(
        key="exp8_kkatiz_clean_image",
        description="kkatiz baseline + clean data + IMAGE-based province prediction",
        base_model="kkatiz/thai-trocr-thaigov-v2",
        model_path=None,
        augment="none",
        use_fp16=True,
        output_name="exp8_kkatiz_clean_image",
    ),
    ImageBasedExperiment(
        key="exp9_kkatiz_aug_image",
        description="kkatiz + heavy augmentation + IMAGE-based province prediction",
        base_model="kkatiz/thai-trocr-thaigov-v2",
        model_path=None,
        augment="heavy",
        use_fp16=True,
        output_name="exp9_kkatiz_aug_image",
    ),
    ImageBasedExperiment(
        key="exp10_openthaigpt_aug_image",
        description="openthaigpt + medium augmentation + IMAGE-based province prediction",
        base_model="openthaigpt/thai-trocr",
        model_path=None,
        augment="medium",
        use_fp16=True,
        output_name="exp10_openthaigpt_aug_image",
    ),
]

EXPERIMENT_LOOKUP = {exp.key: exp for exp in IMAGE_EXPERIMENTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image-based province prediction experiments")
    parser.add_argument("--csv", required=True, help="Path to tb_match_data_*.csv")
    parser.add_argument("--data-root", action="append", required=True, help="Root directory containing the /210/... hierarchy.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[exp.key for exp in IMAGE_EXPERIMENTS],
        help="Keys of experiments to run (default: all image-based experiments).",
    )
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "both"],
        default="both",  # Default to both training and evaluation
        help="Which stage(s) to execute for each experiment.",
    )
    parser.add_argument("--output-root", default="outputs/image_based", help="Directory to store experiment outputs.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples (default: None = use all samples).")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Max eval samples during training (default: None = use all).")
    parser.add_argument("--eval-max-samples", type=int, default=None, help="Max samples during standalone evaluation (default: None = use all).")
    parser.add_argument("--num-train-epochs", type=int, default=5, help="Number of training epochs (increased for better multi-task learning).")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (reduced for stability).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers (0 is Windows-safe).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip training if output dir already has weights.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going even if a command fails.")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU training to avoid CUDA errors.")
    return parser.parse_args()


def build_command(base: Sequence[str], extra: Sequence[str]) -> List[str]:
    return list(base) + list(extra)


def run_command(cmd: Sequence[str], *, dry_run: bool) -> int:
    cmd_display = " ".join(str(x) for x in cmd)
    print(f"\n==> Running: {cmd_display}")
    if dry_run:
        print("[dry-run] command not executed")
        return 0
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        print(f"[error] Command failed with exit code {completed.returncode}")
    return completed.returncode


def has_existing_weights(output_dir: Path) -> bool:
    return (
        (output_dir / "pytorch_model.bin").exists()
        or (output_dir / "model.safetensors").exists()
        or (output_dir / "adapter_model.bin").exists()
        or (output_dir / "adapter_model.safetensors").exists()
        or (output_dir / "adapter_model.pt").exists()
        or (output_dir / "config.json").exists()
    )


def main() -> None:
    args = parse_args()

    selected_experiments = []
    for key in args.experiments:
        if key not in EXPERIMENT_LOOKUP:
            raise SystemExit(f"Unknown experiment key: {key}")
        selected_experiments.append(EXPERIMENT_LOOKUP[key])

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Common training arguments
    train_common: List[str] = [
        "--csv", args.csv,
        "--num-train-epochs", str(args.num_train_epochs),
        "--per-device-train-batch-size", str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
        "--learning-rate", str(args.learning_rate),
    ]
    
    for root in args.data_root:
        train_common.extend(["--data-root", root])

    if args.max_train_samples is not None:
        train_common.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_eval_samples is not None:
        train_common.extend(["--max-eval-samples", str(args.max_eval_samples)])
    if args.cpu_only:
        train_common.extend(["--no-cuda"])

    # Run experiments
    for exp in selected_experiments:
        print(f"\n=== Image-Based Experiment: {exp.key} ===")
        print(f"Description: {exp.description}")
        print(f"Base Model: {exp.base_model}")
        print(f"Model Path: {exp.model_path or 'None (fresh download)'}")
        print(f"Augmentation: {exp.augment}")
        print(f"FP16: {exp.use_fp16}")
        
        out_dir = exp.output_dir(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.stage in ("train", "both"):
            if args.skip_existing and has_existing_weights(out_dir):
                print(f"[skip] Weights already present in {out_dir}, skipping training stage.")
            else:
                # Check if training script exists
                if not TRAIN_SCRIPT.exists():
                    print(f"[error] Training script not found: {TRAIN_SCRIPT}")
                    if not args.continue_on_error:
                        raise SystemExit(1)
                    continue
                    
                train_cmd = build_command(
                    [str(sys.executable), str(TRAIN_SCRIPT)],
                    train_common
                    + ["--output-dir", str(out_dir)]
                    + exp.train_args(),
                )
                code = run_command(train_cmd, dry_run=args.dry_run)
                if code != 0 and not args.continue_on_error:
                    raise SystemExit(code)

        if args.stage in ("eval", "both"):
            # Note: Evaluation might not be implemented yet
            if not EVAL_SCRIPT.exists():
                print(f"[warning] Evaluation script not found: {EVAL_SCRIPT}")
                print("[info] Skipping evaluation - implement eval_multitask_trocr.py later")
                continue
                
            eval_cmd = build_command(
                [str(sys.executable), str(EVAL_SCRIPT)],
                [
                    "--csv", args.csv,
                    "--model-path", str(out_dir),
                ] + exp.eval_args(),
            )
            
            # Only add max-samples if it's not None
            if args.eval_max_samples is not None:
                eval_cmd.extend(["--max-samples", str(args.eval_max_samples)])
                
            for root in args.data_root:
                eval_cmd.extend(["--data-root", root])
                
            save_results = out_dir / "eval_test_image.json"
            eval_cmd.extend(["--save-results", str(save_results)])
            code = run_command(eval_cmd, dry_run=args.dry_run)
            if code != 0 and not args.continue_on_error:
                raise SystemExit(code)

    print("\nAll requested image-based experiments processed.")
    print("\nNext steps:")
    print("1. Check training results in output directories")
    print("2. Implement eval_multitask_trocr.py for evaluation")
    print("3. Compare results with text-based experiments")


if __name__ == "__main__":
    main()