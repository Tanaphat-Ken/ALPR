"""Batch analysis script for multiple TrOCR experiments.

This script analyzes all experiments in the outputs directory and generates
comprehensive performance comparisons.

Usage:
    python train/batch_analyze.py --output-root outputs/grid
    python train/batch_analyze.py --output-root outputs/grid --save-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List, Any
import logging

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install matplotlib pandas numpy")
    exit(1)

from analyze_training import TrainingAnalyzer

logger = logging.getLogger("batch_analyze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch analyze multiple TrOCR experiments")
    parser.add_argument("--output-root", type=str, required=True, help="Root directory containing experiment outputs")
    parser.add_argument("--save-plots", action="store_true", help="Save comparison plots")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Plot format")
    parser.add_argument("--dpi", type=int, default=300, help="Plot resolution")
    return parser.parse_args()


class BatchAnalyzer:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.experiments = {}
        
    def find_experiments(self) -> Dict[str, Path]:
        """Find all experiment directories."""
        experiments = {}
        
        for exp_dir in self.output_root.iterdir():
            if exp_dir.is_dir():
                # Check if it contains training outputs
                trainer_state = exp_dir / "trainer_state.json"
                if trainer_state.exists():
                    experiments[exp_dir.name] = exp_dir
                    logger.info(f"Found experiment: {exp_dir.name}")
        
        return experiments
    
    def extract_final_metrics(self) -> pd.DataFrame:
        """Extract final metrics from all experiments."""
        self.experiments = self.find_experiments()
        
        data = []
        for exp_name, exp_dir in self.experiments.items():
            analyzer = TrainingAnalyzer(exp_dir)
            if analyzer.load_training_logs():
                metrics = analyzer.extract_metrics_over_time()
                
                row = {'experiment': exp_name}
                
                # Final metrics
                if metrics['eval_loss_values']:
                    row['final_val_loss'] = metrics['eval_loss_values'][-1]
                    row['best_val_loss'] = min(metrics['eval_loss_values'])
                
                if metrics['eval_cer_values']:
                    row['final_cer'] = metrics['eval_cer_values'][-1]
                    row['best_cer'] = min(metrics['eval_cer_values'])
                
                if metrics['eval_exact_match_values']:
                    row['final_accuracy'] = metrics['eval_exact_match_values'][-1]
                    row['best_accuracy'] = max(metrics['eval_exact_match_values'])
                
                # Province-specific metrics if available
                if metrics['eval_plate_cer_values']:
                    row['final_plate_cer'] = metrics['eval_plate_cer_values'][-1]
                    row['best_plate_cer'] = min(metrics['eval_plate_cer_values'])
                
                if metrics['eval_province_cer_values']:
                    row['final_province_cer'] = metrics['eval_province_cer_values'][-1]
                    row['best_province_cer'] = min(metrics['eval_province_cer_values'])
                
                if metrics['eval_plate_exact_match_values']:
                    row['final_plate_accuracy'] = metrics['eval_plate_exact_match_values'][-1]
                    row['best_plate_accuracy'] = max(metrics['eval_plate_exact_match_values'])
                
                if metrics['eval_province_exact_match_values']:
                    row['final_province_accuracy'] = metrics['eval_province_exact_match_values'][-1]
                    row['best_province_accuracy'] = max(metrics['eval_province_exact_match_values'])
                
                # Training info
                row['total_steps'] = max(metrics['steps']) if metrics['steps'] else 0
                row['total_epochs'] = max(metrics['epochs']) if metrics['epochs'] else 0
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_comparison(self, df: pd.DataFrame, save_plots: bool = False, 
                       format: str = "png", dpi: int = 300):
        """Create comparison plots between experiments."""
        if df.empty:
            logger.warning("No data to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Best metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experiment Comparison - Best Metrics', fontsize=16)
        
        # Best validation loss
        if 'best_val_loss' in df.columns:
            axes[0, 0].bar(df['experiment'], df['best_val_loss'])
            axes[0, 0].set_title('Best Validation Loss')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Best CER
        if 'best_cer' in df.columns:
            axes[0, 1].bar(df['experiment'], df['best_cer'] * 100)
            axes[0, 1].set_title('Best Character Error Rate')
            axes[0, 1].set_ylabel('CER (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Best accuracy
        if 'best_accuracy' in df.columns:
            axes[1, 0].bar(df['experiment'], df['best_accuracy'] * 100)
            axes[1, 0].set_title('Best Exact Match Accuracy')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training epochs
        if 'total_epochs' in df.columns:
            axes[1, 1].bar(df['experiment'], df['total_epochs'])
            axes[1, 1].set_title('Training Epochs')
            axes[1, 1].set_ylabel('Epochs')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            comparison_dir = self.output_root / "comparison_plots"
            comparison_dir.mkdir(exist_ok=True)
            plt.savefig(comparison_dir / f"best_metrics_comparison.{format}", 
                       format=format, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved comparison plot: {comparison_dir}/best_metrics_comparison.{format}")
            plt.close(fig)
        else:
            plt.show()
        
        # 2. Province-specific comparison (if available)
        province_cols = [col for col in df.columns if 'province' in col or 'plate' in col]
        if province_cols:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Province Prediction Comparison', fontsize=16)
            
            if 'best_plate_cer' in df.columns:
                axes[0, 0].bar(df['experiment'], df['best_plate_cer'] * 100)
                axes[0, 0].set_title('Best Plate CER')
                axes[0, 0].set_ylabel('CER (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            if 'best_province_cer' in df.columns:
                axes[0, 1].bar(df['experiment'], df['best_province_cer'] * 100)
                axes[0, 1].set_title('Best Province CER')
                axes[0, 1].set_ylabel('CER (%)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            if 'best_plate_accuracy' in df.columns:
                axes[1, 0].bar(df['experiment'], df['best_plate_accuracy'] * 100)
                axes[1, 0].set_title('Best Plate Accuracy')
                axes[1, 0].set_ylabel('Accuracy (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            if 'best_province_accuracy' in df.columns:
                axes[1, 1].bar(df['experiment'], df['best_province_accuracy'] * 100)
                axes[1, 1].set_title('Best Province Accuracy')
                axes[1, 1].set_ylabel('Accuracy (%)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(comparison_dir / f"province_metrics_comparison.{format}", 
                           format=format, dpi=dpi, bbox_inches='tight')
                logger.info(f"Saved province comparison: {comparison_dir}/province_metrics_comparison.{format}")
                plt.close(fig)
            else:
                plt.show()
    
    def generate_summary_table(self, df: pd.DataFrame) -> str:
        """Generate a summary table of all experiments."""
        if df.empty:
            return "No experiments found."
        
        # Format numeric columns as percentages where appropriate
        formatted_df = df.copy()
        percentage_cols = [col for col in df.columns if 'cer' in col or 'accuracy' in col]
        for col in percentage_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        
        # Format other numeric columns
        numeric_cols = ['final_val_loss', 'best_val_loss', 'total_steps', 'total_epochs']
        for col in numeric_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and 'loss' in col else f"{x:.1f}" if pd.notna(x) else "N/A")
        
        return formatted_df.to_string(index=False)
    
    def run_analysis(self, save_plots: bool = False, format: str = "png", dpi: int = 300):
        """Run complete batch analysis."""
        logger.info(f"Analyzing experiments in: {self.output_root}")
        
        # Extract metrics
        df = self.extract_final_metrics()
        
        if df.empty:
            logger.warning("No valid experiments found")
            return
        
        # Generate summary
        summary = self.generate_summary_table(df)
        print("\n=== Experiment Comparison Summary ===")
        print(summary)
        
        # Save summary
        if save_plots:
            comparison_dir = self.output_root / "comparison_plots"
            comparison_dir.mkdir(exist_ok=True)
            with open(comparison_dir / "experiment_summary.txt", 'w', encoding='utf-8') as f:
                f.write("=== Experiment Comparison Summary ===\n")
                f.write(summary)
            
            # Also save CSV for further analysis
            df.to_csv(comparison_dir / "experiment_metrics.csv", index=False)
            logger.info(f"Saved summary: {comparison_dir}/experiment_summary.txt")
            logger.info(f"Saved CSV: {comparison_dir}/experiment_metrics.csv")
        
        # Generate comparison plots
        self.plot_comparison(df, save_plots, format, dpi)
        
        # Analyze individual experiments
        for exp_name, exp_dir in self.experiments.items():
            logger.info(f"Analyzing individual experiment: {exp_name}")
            analyzer = TrainingAnalyzer(exp_dir)
            plot_dir = exp_dir / "plots" if save_plots else None
            analyzer.generate_all_plots(
                save_plots=save_plots,
                plot_dir=plot_dir,
                format=format,
                dpi=dpi,
                figsize=(12, 8)
            )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    output_root = Path(args.output_root)
    if not output_root.exists():
        logger.error(f"Output root directory does not exist: {output_root}")
        return
    
    analyzer = BatchAnalyzer(output_root)
    analyzer.run_analysis(
        save_plots=args.save_plots,
        format=args.format,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()