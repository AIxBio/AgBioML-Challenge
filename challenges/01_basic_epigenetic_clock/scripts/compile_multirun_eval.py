#!/usr/bin/env python3
"""
Comprehensive Multirun Analysis Script for BioAgents Experiments

This script analyzes the results from multirun experiments comparing:
- Model performance (gpt-4.1 vs gpt-4o)
- Public evaluation impact (True vs False)
- Resource usage and efficiency metrics

Usage:
    python compile_multirun_eval.py <multirun_output_directory>
    
Example:
    python compile_multirun_eval.py outputs/multirun_comparison_challenge1/2025-05-31/19-38-47
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up publication-quality plotting
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class MultirunAnalyzer:
    """Analyzer for BioAgents multirun experiment results."""
    
    def __init__(self, multirun_dir: Path):
        """Initialize analyzer with multirun directory."""
        self.multirun_dir = Path(multirun_dir)
        self.results_df = None
        self.summary_stats = None
        
        # Model-based color palette with public evaluation as saturation variants
        self.colors = {
            # gpt-4.1 in blue tones
            'gpt-4.1_public-eval-True': '#1f77b4',     # Full blue
            'gpt-4.1_public-eval-False': '#9bb3d9',    # More greyish blue
            # gpt-4o in purple tones  
            'gpt-4o_public-eval-True': '#a855f7',      # Full purple
            'gpt-4o_public-eval-False': '#c8b3e6'      # More greyish purple
        }
        
        # Model colors for consistent theming
        self.model_colors = {
            'gpt-4.1': '#1f77b4',  # Blue
            'gpt-4o': '#a855f7'    # Purple
        }
        
        # Public evaluation colors (for hue mapping)
        self.public_eval_colors = {
            True: 'dark',    # Full saturation
            False: 'light'   # Desaturated/grayed
        }
        
    def discover_run_directories(self) -> List[Path]:
        """Discover all run directories in the multirun output."""
        pattern = re.compile(r'(gpt-[\d\.]+[a-z]*?)_public-eval-(True|False)_run-(\d+)')
        
        run_dirs = []
        for subdir in self.multirun_dir.iterdir():
            if subdir.is_dir() and pattern.match(subdir.name):
                run_dirs.append(subdir)
        
        if not run_dirs:
            raise ValueError(f"No run directories found in {self.multirun_dir}")
            
        print(f"Found {len(run_dirs)} run directories")
        return sorted(run_dirs)
    
    def parse_run_metadata(self, run_dir: Path) -> Dict:
        """Extract metadata from run directory name."""
        pattern = re.compile(r'(gpt-[\d\.]+[a-z]*?)_public-eval-(True|False)_run-(\d+)')
        match = pattern.match(run_dir.name)
        
        if not match:
            raise ValueError(f"Could not parse run directory name: {run_dir.name}")
            
        model, public_eval, run_num = match.groups()
        
        # Convert public_eval to boolean
        public_eval_bool = public_eval == 'True'
        
        # Calculate seed (4 seeds per condition, cycling)
        seed = int(run_num) % 4
        
        return {
            'model': model,
            'public_evaluation': public_eval_bool,
            'run_number': int(run_num),
            'seed': seed,
            'condition': f"{model}_public-eval-{public_eval}",
            'run_dir': str(run_dir)
        }
    
    def load_run_data(self, run_dir: Path) -> Dict:
        """Load data from a single run directory."""
        metadata = self.parse_run_metadata(run_dir)
        
        # Load output stats
        stats_file = run_dir / "output_stats.json"
        stats_data = {}
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
        else:
            print(f"Warning: output_stats.json not found in {run_dir}")
        
        # Load evaluation results
        eval_file = run_dir / "final_eval_results" / "evaluation_results.json"
        eval_data = {}
        has_evaluation_results = False
        
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                has_evaluation_results = True
        else:
            print(f"Warning: evaluation_results.json not found in {run_dir}")
        
        # Combine all data
        combined_data = {**metadata, **stats_data}
        
        # Extract key evaluation metrics with defaults for failed runs
        if has_evaluation_results and 'overall_metrics' in eval_data:
            metrics = eval_data['overall_metrics']
            combined_data.update({
                'pearson_correlation': metrics.get('pearson_correlation'),
                'mae_years': metrics.get('mae_years'),
                'rmse_years': metrics.get('rmse_years'),
                'r_squared': metrics.get('r_squared'),
                'median_ae_years': metrics.get('median_ae_years'),
                'n_samples': metrics.get('n_samples')
            })
        else:
            # Set defaults for failed runs (no evaluation results)
            combined_data.update({
                'pearson_correlation': None,
                'mae_years': None,
                'rmse_years': None,
                'r_squared': None,
                'median_ae_years': None,
                'n_samples': None
            })
        
        # Extract criteria results with defaults for failed runs
        if has_evaluation_results and 'criteria_met' in eval_data:
            combined_data.update({
                'pearson_criteria_met': eval_data['criteria_met'].get('pearson_correlation >= 0.9', False),
                'mae_criteria_met': eval_data['criteria_met'].get('mae_years < 10.0', False),
                'all_criteria_passed': eval_data.get('all_criteria_passed', False)
            })
        else:
            # Failed runs did not meet criteria by definition
            combined_data.update({
                'pearson_criteria_met': False,
                'mae_criteria_met': False,
                'all_criteria_passed': False
            })
        
        # Add a flag to indicate if this run had evaluation results
        combined_data['has_evaluation_results'] = has_evaluation_results
        
        return combined_data
    
    def compile_results(self) -> pd.DataFrame:
        """Compile results from all runs into a DataFrame."""
        run_dirs = self.discover_run_directories()
        
        all_data = []
        for run_dir in run_dirs:
            try:
                run_data = self.load_run_data(run_dir)
                all_data.append(run_data)
            except Exception as e:
                print(f"Error processing {run_dir}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid run data found")
        
        df = pd.DataFrame(all_data)
        
        # Add derived metrics with robust NaN handling
        df['cost_per_token'] = df['estimated_cost_usd'] / df['total_tokens']
        df['tokens_per_minute'] = df['total_tokens'] / df['total_duration_minutes']
        
        # Handle success rate robustly - fill NaN values with False before converting
        df['all_criteria_passed'] = df['all_criteria_passed'].fillna(False)
        df['success_rate'] = df['all_criteria_passed'].astype(int)
        
        # Handle token ratios with potential NaN values
        df['output_input_token_ratio'] = df['total_output_tokens'] / df['total_input_tokens']
        
        # Add completion status indicators
        df['task_completed'] = df['has_evaluation_results']  # True if evaluation results exist
        df['completion_status'] = df.apply(
            lambda row: 'Completed' if row['has_evaluation_results'] else 'Failed',
            axis=1
        )
        
        # Create readable labels for plotting
        df['model_label'] = df['model']  # Keep model as primary grouping
        df['eval_condition'] = df.apply(
            lambda row: f"{row['model']} + public eval" if row['public_evaluation'] else row['model'],
            axis=1
        )
        
        # Create combined condition for backward compatibility and colors
        df['condition'] = df.apply(
            lambda row: f"{row['model']}_public-eval-{row['public_evaluation']}", 
            axis=1
        )
        
        # Create full condition labels for legends/reports
        df['condition_label'] = df.apply(
            lambda row: f"{row['model']} ({'Public' if row['public_evaluation'] else 'Private'} eval)", 
            axis=1
        )
        
        # Print summary of completion status
        completion_summary = df.groupby(['model', 'public_evaluation'])['task_completed'].agg(['sum', 'count']).reset_index()
        print("\n📊 Task Completion Summary:")
        for _, row in completion_summary.iterrows():
            model = row['model']
            public_eval = 'Public eval' if row['public_evaluation'] else 'Private only'
            completed = row['sum']
            total = row['count']
            print(f"  {model} ({public_eval}): {completed}/{total} completed ({completed/total:.1%})")
        
        self.results_df = df
        return df
    
    def calculate_summary_stats(self) -> pd.DataFrame:
        """Calculate summary statistics by condition."""
        if self.results_df is None:
            raise ValueError("Must compile results first")
        
        # Group by full condition label for comprehensive stats
        grouped = self.results_df.groupby('condition_label')
        
        # Calculate stats for key metrics
        metrics = [
            'pearson_correlation', 'mae_years', 'rmse_years',
            'total_input_tokens', 'total_output_tokens', 'total_duration_minutes',
            'estimated_cost_usd', 'iterations_completed', 'success_rate'
        ]
        
        summary_data = []
        for condition, group in grouped:
            condition_stats = {'condition': condition, 'n_runs': len(group)}
            
            for metric in metrics:
                if metric in group.columns and not group[metric].isna().all():
                    condition_stats.update({
                        f'{metric}_mean': group[metric].mean(),
                        f'{metric}_std': group[metric].std(),
                        f'{metric}_min': group[metric].min(),
                        f'{metric}_max': group[metric].max(),
                        f'{metric}_median': group[metric].median()
                    })
            
            summary_data.append(condition_stats)
        
        self.summary_stats = pd.DataFrame(summary_data)
        return self.summary_stats
    
    def create_performance_plots(self, save_dir: Path) -> None:
        """Create performance comparison plots."""
        if self.results_df is None:
            raise ValueError("Must compile results first")
        
        # Define colors: model-based with public eval as full color, private as grayed
        color_mapping = {
            'gpt-4.1': '#9bb3d9',           # More greyish blue for gpt-4.1 (private only)
            'gpt-4.1 + public eval': '#1f77b4',  # Full blue for gpt-4.1 + public eval
            'gpt-4o': '#c8b3e6',            # More greyish purple for gpt-4o (private only)  
            'gpt-4o + public eval': '#a855f7'     # Full purple for gpt-4o + public eval
        }
        
        # Filter to completed runs for performance metrics
        completed_runs = self.results_df[self.results_df['task_completed'] == True]
        
        # Create figure with subplots (2x2 grid for performance only)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold')
        
        # Helper function to create grouped bar plots with error bars
        def create_grouped_errorbar_plot(ax, data, y_col, title, ylabel, target_line=None):
            if data.empty:
                ax.text(0.5, 0.5, 'No completed runs\nwith performance data', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=17)
                ax.set_title(title)
                ax.set_ylabel(ylabel, fontsize=20)
                return
            
            # Calculate stats by model and public evaluation
            models = ['gpt-4.1', 'gpt-4o']
            x_pos = np.arange(len(models))
            width = 0.35
            
            # Get data for each model and evaluation type
            stats = {}
            max_top_value = 0  # Track max value for y-axis limits
            for model in models:
                for public_eval in [True, False]:
                    subset = data[(data['model'] == model) & (data['public_evaluation'] == public_eval)]
                    if not subset.empty and not subset[y_col].isna().all():
                        mean_val = subset[y_col].mean()
                        sem_val = subset[y_col].sem()
                        ci_95 = 1.96 * sem_val
                        top_value = mean_val + ci_95
                        max_top_value = max(max_top_value, top_value)
                        
                        stats[(model, public_eval)] = {
                            'mean': mean_val,
                            'sem': sem_val,
                            'ci_95': ci_95,
                            'data': subset[y_col]
                        }
            
            # Create bars and track legend entries for each model-evaluation combination
            legend_entries = []
            for i, model in enumerate(models):
                # Public eval (full color) - leftmost
                if (model, True) in stats:
                    s = stats[(model, True)]
                    color = color_mapping[f'{model} + public eval']
                    bar1 = ax.bar(i - width/2, s['mean'], width, yerr=s['ci_95'], 
                                 capsize=5, alpha=0.7, color=color)
                    
                    # Add legend entry for this specific model-evaluation combination
                    legend_entries.append((bar1[0], f'{model} + public eval'))
                    
                    # Add jittered points
                    x_jitter = np.random.normal(i - width/2, 0.02, len(s['data']))
                    ax.scatter(x_jitter, s['data'], color='black', alpha=0.6, s=30, zorder=3)
                    
                    # Add value label with avg prefix
                    ax.text(i - width/2, s['mean'] + s['ci_95'] * 1.1, f'avg: {s["mean"]:.3f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=14)
                
                # Private only (grayed color) - rightmost
                if (model, False) in stats:
                    s = stats[(model, False)]
                    color = color_mapping[model]
                    bar2 = ax.bar(i + width/2, s['mean'], width, yerr=s['ci_95'],
                                 capsize=5, alpha=0.7, color=color)
                    
                    # Add legend entry for this specific model-evaluation combination
                    legend_entries.append((bar2[0], f'{model} private only'))
                    
                    # Add jittered points
                    x_jitter = np.random.normal(i + width/2, 0.02, len(s['data']))
                    ax.scatter(x_jitter, s['data'], color='black', alpha=0.6, s=30, zorder=3)
                    
                    # Add value label with avg prefix
                    ax.text(i + width/2, s['mean'] + s['ci_95'] * 1.1, f'avg: {s["mean"]:.3f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=14)
            
            # Add target line if specified
            if target_line is not None:
                ax.axhline(y=target_line['value'], color='red', linestyle='--', alpha=0.7, 
                          label=target_line['label'])
                max_top_value = max(max_top_value, target_line['value'])
            
            # Set y-axis limit to 26% above the maximum value
            if max_top_value > 0:
                ax.set_ylim(bottom=0, top=max_top_value * 1.26)
            
            # Formatting
            ax.set_title(title)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=22)
            
            # Add legend with all model-evaluation combinations
            if legend_entries:
                legend_handles, legend_labels = zip(*legend_entries)
                ax.legend(legend_handles, legend_labels)
            
            ax.grid(True, alpha=0.3)
        
        # Helper function for completion rate (no error bars needed)
        def create_completion_rate_plot(ax, data):
            models = ['gpt-4.1', 'gpt-4o']
            x_pos = np.arange(len(models))
            width = 0.35
            
            # Calculate completion rates
            rates = {}
            max_rate = 0
            for model in models:
                for public_eval in [True, False]:
                    subset = data[(data['model'] == model) & (data['public_evaluation'] == public_eval)]
                    if not subset.empty:
                        rate = subset['task_completed'].mean()
                        rates[(model, public_eval)] = rate
                        max_rate = max(max_rate, rate)
            
            # Create bars and track legend entries for each model-evaluation combination
            legend_entries = []
            for i, model in enumerate(models):
                # Public eval (full color) - leftmost
                if (model, True) in rates:
                    rate = rates[(model, True)]
                    color = color_mapping[f'{model} + public eval']
                    bar1 = ax.bar(i - width/2, rate, width, alpha=0.7, color=color)
                    
                    # Add legend entry for this specific model-evaluation combination
                    legend_entries.append((bar1[0], f'{model} + public eval'))
                    
                    ax.text(i - width/2, rate + max_rate * 0.02, f'avg: {rate:.1%}',
                           ha='center', va='bottom', fontweight='bold')
                
                # Private only (grayed color) - rightmost  
                if (model, False) in rates:
                    rate = rates[(model, False)]
                    color = color_mapping[model]
                    bar2 = ax.bar(i + width/2, rate, width, alpha=0.7, color=color)
                    
                    # Add legend entry for this specific model-evaluation combination
                    legend_entries.append((bar2[0], f'{model} private only'))
                    
                    ax.text(i + width/2, rate + max_rate * 0.02, f'avg: {rate:.1%}',
                           ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Task Completion Rate (All Runs)')
            ax.set_ylabel('Completion Rate (↑)', fontsize=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=22)
            ax.set_ylim(0, max_rate * 1.09)
            
            # Add legend with all model-evaluation combinations
            if legend_entries:
                legend_handles, legend_labels = zip(*legend_entries)
                ax.legend(legend_handles, legend_labels)
            
            ax.grid(True, alpha=0.3)
        
        # Performance metrics (2x2 grid)
        # Pearson Correlation (completed runs only)
        create_grouped_errorbar_plot(axes[0, 0], completed_runs, 'pearson_correlation', 
                                    'Pearson Correlation (Completed Runs Only)', 'Pearson Correlation (↑)',
                                    {'value': 0.9, 'label': 'Target ≥ 0.9'})
        
        # MAE (completed runs only)
        create_grouped_errorbar_plot(axes[0, 1], completed_runs, 'mae_years',
                                    'Mean Absolute Error (Completed Runs Only)', 'MAE in years (↓)',
                                    {'value': 10.0, 'label': 'Target < 10.0'})
        
        # R-squared (completed runs only)
        create_grouped_errorbar_plot(axes[1, 0], completed_runs, 'r_squared',
                                    'R-squared (Completed Runs Only)', 'R² (↑)')
        
        # Task Completion Rate (all runs)
        create_completion_rate_plot(axes[1, 1], self.results_df)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'performance_comparison.pdf', bbox_inches='tight')
        plt.show()
    
    def create_resource_plots(self, save_dir: Path) -> None:
        """Create resource usage and efficiency comparison plots."""
        if self.results_df is None:
            raise ValueError("Must compile results first")
        
        # Define colors: model-based with public eval as full color, private as grayed
        color_mapping = {
            'gpt-4.1': '#9bb3d9',           # More greyish blue for gpt-4.1 (private only)
            'gpt-4.1 + public eval': '#1f77b4',  # Full blue for gpt-4.1 + public eval
            'gpt-4o': '#c8b3e6',            # More greyish purple for gpt-4o (private only)  
            'gpt-4o + public eval': '#a855f7'     # Full purple for gpt-4o + public eval
        }
        
        # Create figure with subplots (2x3 grid for resource + efficiency)
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle('Resource Usage & Efficiency Analysis', fontsize=20, fontweight='bold')
        
        # Helper function to create grouped bar plots for resource metrics
        def create_resource_grouped_plot(ax, data, y_col, title, ylabel, scale_factor=1.0, unit_suffix=""):
            if data.empty or data[y_col].isna().all():
                ax.text(0.5, 0.5, 'No data available', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=17)
                ax.set_title(title)
                ax.set_ylabel(ylabel, fontsize=20)
                return
            
            # Calculate stats by model and public evaluation
            models = ['gpt-4.1', 'gpt-4o']
            x_pos = np.arange(len(models))
            width = 0.35
            
            # Get data for each model and evaluation type
            stats = {}
            max_top_value = 0
            for model in models:
                for public_eval in [True, False]:
                    subset = data[(data['model'] == model) & (data['public_evaluation'] == public_eval)]
                    if not subset.empty and not subset[y_col].isna().all():
                        # Apply scaling factor (e.g., divide by 1M for tokens)
                        scaled_values = subset[y_col] / scale_factor
                        mean_val = scaled_values.mean()
                        sem_val = scaled_values.sem()
                        ci_95 = 1.96 * sem_val
                        top_value = mean_val + ci_95
                        max_top_value = max(max_top_value, top_value)
                        
                        stats[(model, public_eval)] = {
                            'mean': mean_val,
                            'sem': sem_val,
                            'ci_95': ci_95,
                            'data': scaled_values
                        }
            
            # Create bars and track legend entries for each model-evaluation combination
            legend_entries = []
            for i, model in enumerate(models):
                # Public eval (full color) - leftmost
                if (model, True) in stats:
                    s = stats[(model, True)]
                    color = color_mapping[f'{model} + public eval']
                    bar1 = ax.bar(i - width/2, s['mean'], width, yerr=s['ci_95'], 
                                 capsize=5, alpha=0.7, color=color)
                    
                    # Add legend entry for this specific model-evaluation combination
                    legend_entries.append((bar1[0], f'{model} + public eval'))
                    
                    # Add jittered points
                    x_jitter = np.random.normal(i - width/2, 0.02, len(s['data']))
                    ax.scatter(x_jitter, s['data'], color='black', alpha=0.6, s=30, zorder=3)
                    
                    # Add value label with appropriate formatting
                    if 'cost' in y_col.lower():
                        label = f'avg: ${s["mean"]:.2f}'
                    elif 'token' in y_col.lower() and 'ratio' not in y_col.lower():
                        label = f'avg: {s["mean"]:.1f}{unit_suffix}'
                    elif 'minute' in y_col.lower():
                        label = f'avg: {s["mean"]:.0f}m'
                    elif 'iteration' in y_col.lower():
                        label = f'avg: {s["mean"]:.1f}'
                    elif 'ratio' in y_col.lower():
                        label = f'avg: {s["mean"]:.2f}'
                    else:
                        label = f'avg: {s["mean"]:.1f}'
                    ax.text(i - width/2, s['mean'] + s['ci_95'] * 1.1, label,
                           ha='center', va='bottom', fontweight='bold', fontsize=14)
                
                # Private only (grayed color) - rightmost
                if (model, False) in stats:
                    s = stats[(model, False)]
                    color = color_mapping[model]
                    bar2 = ax.bar(i + width/2, s['mean'], width, yerr=s['ci_95'],
                                 capsize=5, alpha=0.7, color=color)
                    
                    # Add legend entry for this specific model-evaluation combination
                    legend_entries.append((bar2[0], f'{model} private only'))
                    
                    # Add jittered points
                    x_jitter = np.random.normal(i + width/2, 0.02, len(s['data']))
                    ax.scatter(x_jitter, s['data'], color='black', alpha=0.6, s=30, zorder=3)
                    
                    # Add value label with appropriate formatting
                    if 'cost' in y_col.lower():
                        label = f'avg: ${s["mean"]:.2f}'
                    elif 'token' in y_col.lower() and 'ratio' not in y_col.lower():
                        label = f'avg: {s["mean"]:.1f}{unit_suffix}'
                    elif 'minute' in y_col.lower():
                        label = f'avg: {s["mean"]:.0f}m'
                    elif 'iteration' in y_col.lower():
                        label = f'avg: {s["mean"]:.1f}'
                    elif 'ratio' in y_col.lower():
                        label = f'avg: {s["mean"]:.2f}'
                    else:
                        label = f'avg: {s["mean"]:.1f}'
                    ax.text(i + width/2, s['mean'] + s['ci_95'] * 1.1, label,
                           ha='center', va='bottom', fontweight='bold', fontsize=14)
            
            # Set y-axis limit to 20% above the maximum value
            if max_top_value > 0:
                ax.set_ylim(bottom=0, top=max_top_value * 1.20)
            
            # Formatting
            ax.set_title(title)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=22)
            
            # Add legend with all model-evaluation combinations
            if legend_entries:
                legend_handles, legend_labels = zip(*legend_entries)
                ax.legend(legend_handles, legend_labels)
            
            ax.grid(True, alpha=0.3)
        
        # Row 1: Resource usage metrics
        # Input Tokens (in millions)
        create_resource_grouped_plot(axes[0, 0], self.results_df, 'total_input_tokens',
                                   'Total Input Tokens', 'Input Tokens in Millions (↓)', 
                                   scale_factor=1_000_000, unit_suffix="M")
        
        # Output Tokens (in millions)
        create_resource_grouped_plot(axes[0, 1], self.results_df, 'total_output_tokens',
                                   'Total Output Tokens', 'Output Tokens in Millions (↓)', 
                                   scale_factor=1_000_000, unit_suffix="M")
        
        # Runtime
        create_resource_grouped_plot(axes[0, 2], self.results_df, 'total_duration_minutes',
                                   'Runtime (Minutes)', 'Duration in minutes (↓)')
        
        # Row 2: Cost and efficiency metrics
        # Estimated Cost
        create_resource_grouped_plot(axes[1, 0], self.results_df, 'estimated_cost_usd',
                                   'Estimated Cost (USD)', 'Cost in USD (↓)')
        
        # Iterations Completed (efficiency)
        create_resource_grouped_plot(axes[1, 1], self.results_df, 'iterations_completed',
                                   'Iterations to Completion', 'Iterations (↓)')
        
        # Token Efficiency (Output/Input ratio)
        create_resource_grouped_plot(axes[1, 2], self.results_df, 'output_input_token_ratio',
                                   'Token Efficiency (Output/Input)', 'Output/Input Token Ratio (↑)')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'resource_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'resource_efficiency_analysis.pdf', bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, save_dir: Path) -> None:
        """Create a comprehensive summary report."""
        if self.results_df is None or self.summary_stats is None:
            raise ValueError("Must compile results and calculate summary stats first")
        
        report_lines = [
            "# BioAgents Multirun Experiment Analysis Report",
            f"Generated from: {self.multirun_dir}",
            f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Experiment Overview",
            f"- Total runs analyzed: {len(self.results_df)}",
            f"- Models compared: {', '.join(self.results_df['model'].unique())}",
            f"- Public evaluation settings: {', '.join(map(str, self.results_df['public_evaluation'].unique()))}",
            f"- Seeds per condition: {len(self.results_df) // len(self.results_df['condition_label'].unique())}",
            "",
            "## Task Completion Analysis",
        ]
        
        # Add completion rate analysis
        completion_analysis = self.results_df.groupby('condition_label').agg({
            'task_completed': ['sum', 'count', 'mean'],
            'completion_reason': lambda x: x.value_counts().to_dict()
        }).round(3)
        
        for condition in completion_analysis.index:
            completed = completion_analysis.loc[condition, ('task_completed', 'sum')]
            total = completion_analysis.loc[condition, ('task_completed', 'count')]
            rate = completion_analysis.loc[condition, ('task_completed', 'mean')]
            reasons = completion_analysis.loc[condition, ('completion_reason', '<lambda>')]
            
            report_lines.extend([
                f"\n### {condition}",
                f"- Completion rate: {completed}/{total} ({rate:.1%})",
                f"- Completion reasons: {reasons}"
            ])
        
        # Performance analysis (only for completed runs)
        completed_runs = self.results_df[self.results_df['task_completed'] == True]
        
        if not completed_runs.empty:
            report_lines.extend([
                "",
                "## Performance Summary (Completed Runs Only)",
            ])
            
            # Add performance statistics for completed runs only
            for condition in completed_runs['condition_label'].unique():
                condition_data = completed_runs[completed_runs['condition_label'] == condition]
                if not condition_data.empty:
                    report_lines.extend([
                        f"\n### {condition}",
                        f"- Completed runs: {len(condition_data)}",
                        f"- Pearson correlation: {condition_data['pearson_correlation'].mean():.3f} ± {condition_data['pearson_correlation'].std():.3f}",
                        f"- MAE (years): {condition_data['mae_years'].mean():.2f} ± {condition_data['mae_years'].std():.2f}",
                        f"- Success rate (criteria met): {condition_data['success_rate'].mean():.1%}",
                        f"- Avg cost: ${condition_data['estimated_cost_usd'].mean():.2f} ± ${condition_data['estimated_cost_usd'].std():.2f}",
                        f"- Avg runtime: {condition_data['total_duration_minutes'].mean():.1f} ± {condition_data['total_duration_minutes'].std():.1f} minutes"
                    ])
        else:
            report_lines.extend([
                "",
                "## Performance Summary",
                "⚠️ No runs completed successfully - no performance metrics available"
            ])
        
        # Resource usage analysis (all runs)
        report_lines.extend([
            "",
            "## Resource Usage Analysis (All Runs)",
        ])
        
        for condition in self.results_df['condition_label'].unique():
            condition_data = self.results_df[self.results_df['condition_label'] == condition]
            report_lines.extend([
                f"\n### {condition}",
                f"- Total runs: {len(condition_data)}",
                f"- Avg tokens: {condition_data['total_tokens'].mean():.0f} ± {condition_data['total_tokens'].std():.0f}",
                f"- Avg cost: ${condition_data['estimated_cost_usd'].mean():.2f} ± ${condition_data['estimated_cost_usd'].std():.2f}",
                f"- Avg runtime: {condition_data['total_duration_minutes'].mean():.1f} ± {condition_data['total_duration_minutes'].std():.1f} minutes",
                f"- Avg iterations: {condition_data['iterations_completed'].mean():.1f} ± {condition_data['iterations_completed'].std():.1f}"
            ])
        
        # Statistical comparisons
        report_lines.extend([
            "",
            "## Key Findings",
            "### Model Comparison (gpt-4.1 vs gpt-4o)",
        ])
        
        # Compare models across all runs
        gpt41_data = self.results_df[self.results_df['model'] == 'gpt-4.1']
        gpt4o_data = self.results_df[self.results_df['model'] == 'gpt-4o']
        
        if not gpt41_data.empty and not gpt4o_data.empty:
            report_lines.extend([
                f"- gpt-4.1 completion rate: {gpt41_data['task_completed'].mean():.1%}",
                f"- gpt-4o completion rate: {gpt4o_data['task_completed'].mean():.1%}",
                f"- gpt-4.1 avg cost: ${gpt41_data['estimated_cost_usd'].mean():.2f}",
                f"- gpt-4o avg cost: ${gpt4o_data['estimated_cost_usd'].mean():.2f}",
                f"- gpt-4.1 avg runtime: {gpt41_data['total_duration_minutes'].mean():.1f} min",
                f"- gpt-4o avg runtime: {gpt4o_data['total_duration_minutes'].mean():.1f} min",
            ])
            
            # Performance comparison (only for completed runs)
            gpt41_completed = gpt41_data[gpt41_data['task_completed'] == True]
            gpt4o_completed = gpt4o_data[gpt4o_data['task_completed'] == True]
            
            if not gpt41_completed.empty and not gpt4o_completed.empty:
                report_lines.extend([
                    "",
                    "**Performance (completed runs only):**",
                    f"- gpt-4.1 avg Pearson: {gpt41_completed['pearson_correlation'].mean():.3f}",
                    f"- gpt-4o avg Pearson: {gpt4o_completed['pearson_correlation'].mean():.3f}",
                    f"- gpt-4.1 avg MAE: {gpt41_completed['mae_years'].mean():.2f} years",
                    f"- gpt-4o avg MAE: {gpt4o_completed['mae_years'].mean():.2f} years",
                ])
        
        # Public evaluation impact
        report_lines.extend([
            "",
            "### Public Evaluation Impact",
        ])
        
        public_data = self.results_df[self.results_df['public_evaluation'] == True]
        private_data = self.results_df[self.results_df['public_evaluation'] == False]
        
        if not public_data.empty and not private_data.empty:
            report_lines.extend([
                f"- With public eval completion rate: {public_data['task_completed'].mean():.1%}",
                f"- Without public eval completion rate: {private_data['task_completed'].mean():.1%}",
                f"- With public eval avg cost: ${public_data['estimated_cost_usd'].mean():.2f}",
                f"- Without public eval avg cost: ${private_data['estimated_cost_usd'].mean():.2f}",
                f"- With public eval avg runtime: {public_data['total_duration_minutes'].mean():.1f} min",
                f"- Without public eval avg runtime: {private_data['total_duration_minutes'].mean():.1f} min",
            ])
            
            # Performance comparison (only for completed runs)
            public_completed = public_data[public_data['task_completed'] == True]
            private_completed = private_data[private_data['task_completed'] == True]
            
            if not public_completed.empty and not private_completed.empty:
                report_lines.extend([
                    "",
                    "**Performance (completed runs only):**",
                    f"- With public eval avg Pearson: {public_completed['pearson_correlation'].mean():.3f}",
                    f"- Without public eval avg Pearson: {private_completed['pearson_correlation'].mean():.3f}",
                    f"- With public eval avg MAE: {public_completed['mae_years'].mean():.2f} years",
                    f"- Without public eval avg MAE: {private_completed['mae_years'].mean():.2f} years",
                ])
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(save_dir / 'analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print("Analysis Report:")
        print("=" * 50)
        print(report_text)
    
    def save_data(self, save_dir: Path) -> None:
        """Save compiled data and summary statistics."""
        if self.results_df is not None:
            self.results_df.to_csv(save_dir / 'compiled_results.csv', index=False)
            
        if self.summary_stats is not None:
            self.summary_stats.to_csv(save_dir / 'summary_statistics.csv', index=False)
    
    def run_full_analysis(self, output_dir: Optional[Path] = None) -> None:
        """Run the complete analysis pipeline."""
        if output_dir is None:
            output_dir = self.multirun_dir / 'analysis_results'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("🔍 Compiling results...")
        self.compile_results()
        
        print("📊 Calculating summary statistics...")
        self.calculate_summary_stats()
        
        print("📈 Creating performance plots...")
        self.create_performance_plots(output_dir)
        
        print("⚡ Creating resource usage & efficiency plots...")
        self.create_resource_plots(output_dir)
        
        print("📝 Generating summary report...")
        self.create_summary_report(output_dir)
        
        print("💾 Saving data files...")
        self.save_data(output_dir)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        print(f"📊 View plots: {output_dir}/*.png")
        print(f"   - performance_comparison.png: Model performance metrics")
        print(f"   - resource_efficiency_analysis.png: Resource usage & efficiency metrics")
        print(f"📄 Read report: {output_dir}/analysis_report.md")
        print(f"📋 Data files: {output_dir}/*.csv")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze BioAgents multirun experiment results')
    parser.add_argument('multirun_dir', type=str, 
                       help='Path to multirun output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis results (default: multirun_dir/analysis_results)')
    
    args = parser.parse_args()
    
    multirun_path = Path(args.multirun_dir)
    if not multirun_path.exists():
        raise ValueError(f"Multirun directory does not exist: {multirun_path}")
    
    output_path = Path(args.output_dir) if args.output_dir else None
    
    # Run analysis
    analyzer = MultirunAnalyzer(multirun_path)
    analyzer.run_full_analysis(output_path)


if __name__ == "__main__":
    main()
