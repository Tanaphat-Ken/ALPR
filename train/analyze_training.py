"""Training analysis and visualization script for TrOCR experiments.

This script analyzes training logs and generates comprehensive performance graphs including:
- Train/validation loss curves
- Character Error Rate (CER) progression
- Exact match accuracy over time
- Learning rate scheduling
- Province prediction metrics (if applicable)

Usage:
    python train/analyze_training.py --output-dir outputs/exp1_v1_clean
    python train/analyze_training.py --output-dir outputs/grid/exp1_v1_clean --save-plots
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import numpy as np
    import pandas as pd
    from datetime import datetime
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install matplotlib pandas numpy")
    exit(1)

logger = logging.getLogger("analyze_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TrOCR training results and generate performance graphs")
    parser.add_argument("--output-dir", type=str, required=True, help="Training output directory containing logs")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files instead of showing")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save plots (default: output-dir/plots)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Plot format")
    parser.add_argument("--dpi", type=int, default=300, help="Plot resolution")
    parser.add_argument("--figsize", type=str, default="12,8", help="Figure size as 'width,height'")
    return parser.parse_args()


class TrainingAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logs_data = []
        self.metrics_data = {}
        self.has_province_metrics = False
        
    def load_training_logs(self) -> bool:
        """Load training logs from various sources."""
        # Try to find training logs
        log_files = [
            self.output_dir / "trainer_state.json",
            self.output_dir / "training_args.json",
            self.output_dir / "val_metrics.json",
            self.output_dir / "test_metrics.json"
        ]
        
        # Load trainer state (main source of training metrics)
        trainer_state_file = self.output_dir / "trainer_state.json"
        
        # If not found in main directory, look in the latest checkpoint
        if not trainer_state_file.exists():
            # Find the latest checkpoint directory
            checkpoint_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
            if checkpoint_dirs:
                # Sort by checkpoint number to get the latest
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split('-')[1]))
                trainer_state_file = latest_checkpoint / "trainer_state.json"
                logger.info(f"Using trainer_state.json from {latest_checkpoint.name}")
        
        if trainer_state_file.exists():
            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
                self.logs_data = trainer_state.get("log_history", [])
                logger.info(f"Loaded {len(self.logs_data)} log entries from trainer_state.json")
        else:
            logger.warning("trainer_state.json not found - limited analysis possible")
            
        # Load final metrics
        for metrics_file in ["val_metrics.json", "test_metrics.json"]:
            metrics_path = self.output_dir / metrics_file
            if metrics_path.exists():
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.metrics_data[metrics_file.replace('.json', '')] = json.load(f)
                    
        # Check if we have province prediction metrics
        self.has_province_metrics = any(
            any("province" in key for key in entry.keys()) 
            for entry in self.logs_data
        )
        
        return len(self.logs_data) > 0
    
    def extract_metrics_over_time(self) -> Dict[str, List]:
        """Extract training metrics organized by time/step."""
        metrics = {
            'steps': [],
            'epochs': [],
            'train_loss': [],
            'eval_loss': [],
            'eval_cer': [],
            'eval_exact_match': [],
            'learning_rate': [],
            'eval_plate_cer': [],
            'eval_plate_exact_match': [],
            'eval_province_cer': [],
            'eval_province_exact_match': []
        }
        
        for entry in self.logs_data:
            if 'step' in entry:
                metrics['steps'].append(entry['step'])
                metrics['epochs'].append(entry.get('epoch', 0))
                
                # Training loss
                if 'train_loss' in entry:
                    metrics['train_loss'].append(entry['train_loss'])
                else:
                    metrics['train_loss'].append(None)
                    
                # Evaluation metrics
                if 'eval_loss' in entry:
                    metrics['eval_loss'].append(entry['eval_loss'])
                    metrics['eval_cer'].append(entry.get('eval_cer', None))
                    metrics['eval_exact_match'].append(entry.get('eval_exact_match', None))
                    
                    # Province-specific metrics
                    if self.has_province_metrics:
                        metrics['eval_plate_cer'].append(entry.get('eval_plate_cer', None))
                        metrics['eval_plate_exact_match'].append(entry.get('eval_plate_exact_match', None))
                        metrics['eval_province_cer'].append(entry.get('eval_province_cer', None))
                        metrics['eval_province_exact_match'].append(entry.get('eval_province_exact_match', None))
                else:
                    for key in ['eval_loss', 'eval_cer', 'eval_exact_match', 
                               'eval_plate_cer', 'eval_plate_exact_match',
                               'eval_province_cer', 'eval_province_exact_match']:
                        metrics[key].append(None)
                
                # Learning rate
                metrics['learning_rate'].append(entry.get('learning_rate', None))
        
        # Remove None values for cleaner plotting
        keys_to_process = list(metrics.keys())  # Create a copy of keys
        for key in keys_to_process:
            if key in ['steps', 'epochs']:
                continue
            # Keep only non-None values with corresponding steps
            filtered_data = [(step, val) for step, val in zip(metrics['steps'], metrics[key]) if val is not None]
            if filtered_data:
                steps, values = zip(*filtered_data)
                metrics[f'{key}_steps'] = list(steps)
                metrics[f'{key}_values'] = list(values)
            else:
                metrics[f'{key}_steps'] = []
                metrics[f'{key}_values'] = []
        
        return metrics
    
    def plot_loss_curves(self, metrics: Dict, figsize: Tuple[int, int]) -> Figure:
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Training loss
        if metrics['train_loss_values']:
            ax.plot(metrics['train_loss_steps'], metrics['train_loss_values'], 
                   label='Training Loss', color='blue', alpha=0.7)
        
        # Validation loss
        if metrics['eval_loss_values']:
            ax.plot(metrics['eval_loss_steps'], metrics['eval_loss_values'],
                   label='Validation Loss', color='red', alpha=0.7, marker='o', markersize=4)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add epoch markers on top x-axis if available
        if metrics['epochs'] and len(set(metrics['epochs'])) > 1:
            ax2 = ax.twiny()
            epoch_steps = []
            epoch_labels = []
            current_epoch = -1
            for step, epoch in zip(metrics['steps'], metrics['epochs']):
                if epoch != current_epoch:
                    epoch_steps.append(step)
                    epoch_labels.append(f'E{int(epoch)}')
                    current_epoch = epoch
            
            if epoch_steps:
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(epoch_steps[:10])  # Limit to avoid crowding
                ax2.set_xticklabels(epoch_labels[:10])
                ax2.set_xlabel('Epochs')
        
        plt.tight_layout()
        return fig
    
    def plot_cer_progression(self, metrics: Dict, figsize: Tuple[int, int]) -> Figure:
        """Plot Character Error Rate progression."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.has_province_metrics:
            # Separate CER for plate and province
            if metrics['eval_plate_cer_values']:
                ax.plot(metrics['eval_plate_cer_steps'], metrics['eval_plate_cer_values'],
                       label='Plate CER', color='blue', marker='o', markersize=4)
            
            if metrics['eval_province_cer_values']:
                ax.plot(metrics['eval_province_cer_steps'], metrics['eval_province_cer_values'],
                       label='Province CER', color='green', marker='s', markersize=4)
            
            if metrics['eval_cer_values']:
                ax.plot(metrics['eval_cer_steps'], metrics['eval_cer_values'],
                       label='Combined CER', color='red', marker='^', markersize=4)
        else:
            # Standard CER
            if metrics['eval_cer_values']:
                ax.plot(metrics['eval_cer_steps'], metrics['eval_cer_values'],
                       label='CER', color='blue', marker='o', markersize=4)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Character Error Rate')
        ax.set_title('Character Error Rate Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        return fig
    
    def plot_accuracy_progression(self, metrics: Dict, figsize: Tuple[int, int]) -> Figure:
        """Plot exact match accuracy progression."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.has_province_metrics:
            # Separate accuracy for plate and province
            if metrics['eval_plate_exact_match_values']:
                ax.plot(metrics['eval_plate_exact_match_steps'], metrics['eval_plate_exact_match_values'],
                       label='Plate Accuracy', color='blue', marker='o', markersize=4)
            
            if metrics['eval_province_exact_match_values']:
                ax.plot(metrics['eval_province_exact_match_steps'], metrics['eval_province_exact_match_values'],
                       label='Province Accuracy', color='green', marker='s', markersize=4)
            
            if metrics['eval_exact_match_values']:
                ax.plot(metrics['eval_exact_match_steps'], metrics['eval_exact_match_values'],
                       label='Combined Accuracy', color='red', marker='^', markersize=4)
        else:
            # Standard accuracy
            if metrics['eval_exact_match_values']:
                ax.plot(metrics['eval_exact_match_steps'], metrics['eval_exact_match_values'],
                       label='Exact Match Accuracy', color='blue', marker='o', markersize=4)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Exact Match Accuracy')
        ax.set_title('Exact Match Accuracy Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        return fig
    
    def plot_learning_rate(self, metrics: Dict, figsize: Tuple[int, int]) -> Figure:
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if metrics['learning_rate_values']:
            ax.plot(metrics['learning_rate_steps'], metrics['learning_rate_values'],
                   label='Learning Rate', color='purple', linewidth=2)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, metrics: Dict) -> str:
        """Create a text summary of training results."""
        report = []
        report.append("=== Training Analysis Summary ===\n")
        
        # Basic info
        total_steps = max(metrics['steps']) if metrics['steps'] else 0
        total_epochs = max(metrics['epochs']) if metrics['epochs'] else 0
        report.append(f"Total Steps: {total_steps}")
        report.append(f"Total Epochs: {total_epochs:.2f}")
        
        # Final metrics
        if metrics['train_loss_values']:
            final_train_loss = metrics['train_loss_values'][-1]
            report.append(f"Final Training Loss: {final_train_loss:.4f}")
        
        if metrics['eval_loss_values']:
            final_eval_loss = metrics['eval_loss_values'][-1]
            best_eval_loss = min(metrics['eval_loss_values'])
            report.append(f"Final Validation Loss: {final_eval_loss:.4f}")
            report.append(f"Best Validation Loss: {best_eval_loss:.4f}")
        
        if metrics['eval_cer_values']:
            final_cer = metrics['eval_cer_values'][-1]
            best_cer = min(metrics['eval_cer_values'])
            report.append(f"Final CER: {final_cer:.2%}")
            report.append(f"Best CER: {best_cer:.2%}")
        
        if metrics['eval_exact_match_values']:
            final_acc = metrics['eval_exact_match_values'][-1]
            best_acc = max(metrics['eval_exact_match_values'])
            report.append(f"Final Accuracy: {final_acc:.2%}")
            report.append(f"Best Accuracy: {best_acc:.2%}")
        
        # Province-specific metrics
        if self.has_province_metrics:
            report.append("\n=== Province Prediction Metrics ===")
            
            if metrics['eval_plate_cer_values']:
                final_plate_cer = metrics['eval_plate_cer_values'][-1]
                best_plate_cer = min(metrics['eval_plate_cer_values'])
                report.append(f"Final Plate CER: {final_plate_cer:.2%}")
                report.append(f"Best Plate CER: {best_plate_cer:.2%}")
            
            if metrics['eval_province_cer_values']:
                final_prov_cer = metrics['eval_province_cer_values'][-1]
                best_prov_cer = min(metrics['eval_province_cer_values'])
                report.append(f"Final Province CER: {final_prov_cer:.2%}")
                report.append(f"Best Province CER: {best_prov_cer:.2%}")
            
            if metrics['eval_plate_exact_match_values']:
                final_plate_acc = metrics['eval_plate_exact_match_values'][-1]
                best_plate_acc = max(metrics['eval_plate_exact_match_values'])
                report.append(f"Final Plate Accuracy: {final_plate_acc:.2%}")
                report.append(f"Best Plate Accuracy: {best_plate_acc:.2%}")
            
            if metrics['eval_province_exact_match_values']:
                final_prov_acc = metrics['eval_province_exact_match_values'][-1]
                best_prov_acc = max(metrics['eval_province_exact_match_values'])
                report.append(f"Final Province Accuracy: {final_prov_acc:.2%}")
                report.append(f"Best Province Accuracy: {best_prov_acc:.2%}")
        
        # Final metrics from saved files
        if 'val_metrics' in self.metrics_data:
            report.append("\n=== Final Validation Metrics ===")
            val_metrics = self.metrics_data['val_metrics']
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    if 'cer' in key.lower() or 'accuracy' in key.lower() or 'exact_match' in key.lower():
                        report.append(f"{key}: {value:.2%}")
                    else:
                        report.append(f"{key}: {value:.4f}")
        
        if 'test_metrics' in self.metrics_data:
            report.append("\n=== Final Test Metrics ===")
            test_metrics = self.metrics_data['test_metrics']
            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    if 'cer' in key.lower() or 'accuracy' in key.lower() or 'exact_match' in key.lower():
                        report.append(f"{key}: {value:.2%}")
                    else:
                        report.append(f"{key}: {value:.4f}")
        
        return "\n".join(report)
    
    def generate_all_plots(self, save_plots: bool = False, plot_dir: Optional[Path] = None, 
                          format: str = "png", dpi: int = 300, figsize: Tuple[int, int] = (12, 8)):
        """Generate all training analysis plots."""
        if not self.load_training_logs():
            logger.error("Could not load training logs")
            return
        
        metrics = self.extract_metrics_over_time()
        
        if save_plots and plot_dir:
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate summary report
        summary = self.create_summary_report(metrics)
        print(summary)
        
        if save_plots and plot_dir:
            with open(plot_dir / "training_summary.txt", 'w', encoding='utf-8') as f:
                f.write(summary)
        
        # Generate plots
        plots = []
        
        # 1. Loss curves
        if metrics['train_loss_values'] or metrics['eval_loss_values']:
            fig = self.plot_loss_curves(metrics, figsize)
            plots.append(("loss_curves", fig))
        
        # 2. CER progression
        if (metrics['eval_cer_values'] or metrics['eval_plate_cer_values'] or 
            metrics['eval_province_cer_values']):
            fig = self.plot_cer_progression(metrics, figsize)
            plots.append(("cer_progression", fig))
        
        # 3. Accuracy progression
        if (metrics['eval_exact_match_values'] or metrics['eval_plate_exact_match_values'] or 
            metrics['eval_province_exact_match_values']):
            fig = self.plot_accuracy_progression(metrics, figsize)
            plots.append(("accuracy_progression", fig))
        
        # 4. Learning rate
        if metrics['learning_rate_values']:
            fig = self.plot_learning_rate(metrics, figsize)
            plots.append(("learning_rate", fig))
        
        # Save or show plots
        if save_plots and plot_dir:
            for name, fig in plots:
                filepath = plot_dir / f"{name}.{format}"
                fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
                logger.info(f"Saved plot: {filepath}")
                plt.close(fig)
        else:
            for name, fig in plots:
                fig.suptitle(f"{name.replace('_', ' ').title()} - {self.output_dir.name}")
                plt.show()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    plot_dir = None
    if args.save_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots"
    
    figsize = tuple(map(int, args.figsize.split(',')))
    
    analyzer = TrainingAnalyzer(output_dir)
    analyzer.generate_all_plots(
        save_plots=args.save_plots,
        plot_dir=plot_dir,
        format=args.format,
        dpi=args.dpi,
        figsize=figsize
    )


if __name__ == "__main__":
    main()