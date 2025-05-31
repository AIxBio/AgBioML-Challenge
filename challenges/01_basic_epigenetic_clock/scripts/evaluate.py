#!/usr/bin/env python
"""Final evaluation script for the Basic Epigenetic Clock challenge.

This script performs the FINAL evaluation using ONLY the private test set
to prevent overfitting to the public test set that agents can access during development.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for high-quality plots
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.2
})


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load predictions from various possible formats.
    
    Args:
        predictions_path: Path to predictions file
        
    Returns:
        DataFrame with 'sample_id' and 'predicted_age' columns
    """
    if predictions_path.suffix == '.arrow':
        return pd.read_feather(predictions_path)
    elif predictions_path.suffix == '.csv':
        return pd.read_csv(predictions_path)
    elif predictions_path.suffix == '.json':
        with open(predictions_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {predictions_path.suffix}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for age prediction.
    
    Args:
        y_true: True ages
        y_pred: Predicted ages
        
    Returns:
        Dictionary of metrics
    """
    # Pearson correlation
    corr, p_value = pearsonr(y_true, y_pred)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Median Absolute Error
    median_ae = np.median(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'pearson_correlation': float(corr),
        'pearson_p_value': float(p_value),
        'mae_years': float(mae),
        'rmse_years': float(rmse),
        'median_ae_years': float(median_ae),
        'r_squared': float(r_squared),
        'n_samples': len(y_true)
    }


def create_publication_plots(merged_data: pd.DataFrame, output_dir: Path) -> None:
    """
    Create beautiful, publication-ready scatter plots of predicted vs true ages.
    
    Args:
        merged_data: DataFrame with 'age', 'predicted_age', and metadata columns
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Epigenetic Age Prediction Performance', fontsize=18, fontweight='bold', y=0.98)
    
    # Define color palettes for different categories
    dataset_palette = sns.color_palette("Set2", n_colors=merged_data.get('dataset', pd.Series()).nunique() or 1)
    gender_palette = {'Male': '#1f77b4', 'Female': '#ff7f0e', 'M': '#1f77b4', 'F': '#ff7f0e'}
    tissue_palette = sns.color_palette("tab10", n_colors=merged_data.get('tissue_type', pd.Series()).nunique() or 1)
    
    # Calculate overall correlation and MAE
    overall_corr, overall_p = pearsonr(merged_data['age'], merged_data['predicted_age'])
    overall_mae = mean_absolute_error(merged_data['age'], merged_data['predicted_age'])
    
    # Common plot settings
    min_age = min(merged_data['age'].min(), merged_data['predicted_age'].min())
    max_age = max(merged_data['age'].max(), merged_data['predicted_age'].max())
    age_range = [min_age - 2, max_age + 2]
    
    # Plot 1: Colored by dataset (top-left)
    ax1 = axes[0, 0]
    if 'dataset' in merged_data.columns and merged_data['dataset'].nunique() > 1:
        for i, dataset in enumerate(merged_data['dataset'].unique()):
            data_subset = merged_data[merged_data['dataset'] == dataset]
            ax1.scatter(data_subset['age'], data_subset['predicted_age'], 
                       alpha=0.7, s=50, color=dataset_palette[i], label=dataset, edgecolors='white', linewidth=0.5)
        ax1.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        title1 = 'Colored by Dataset'
    else:
        ax1.scatter(merged_data['age'], merged_data['predicted_age'], 
                   alpha=0.7, s=50, color='#2E86AB', edgecolors='white', linewidth=0.5)
        title1 = 'All Samples'
    
    ax1.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Chronological Age (years)', fontweight='bold')
    ax1.set_ylabel('Predicted Age (years)', fontweight='bold')
    ax1.set_title(title1, fontweight='bold', pad=20)
    ax1.set_xlim(age_range)
    ax1.set_ylim(age_range)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation and MAE text
    ax1.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Plot 2: Colored by gender (top-right)
    ax2 = axes[0, 1]
    if 'gender' in merged_data.columns and merged_data['gender'].nunique() > 1:
        for gender in merged_data['gender'].unique():
            if pd.notna(gender):
                data_subset = merged_data[merged_data['gender'] == gender]
                color = gender_palette.get(gender, '#2E86AB')
                ax2.scatter(data_subset['age'], data_subset['predicted_age'], 
                           alpha=0.7, s=50, color=color, label=gender, edgecolors='white', linewidth=0.5)
        ax2.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        title2 = 'Colored by Gender'
    else:
        ax2.scatter(merged_data['age'], merged_data['predicted_age'], 
                   alpha=0.7, s=50, color='#A23B72', edgecolors='white', linewidth=0.5)
        title2 = 'All Samples'
    
    ax2.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Chronological Age (years)', fontweight='bold')
    ax2.set_ylabel('Predicted Age (years)', fontweight='bold')
    ax2.set_title(title2, fontweight='bold', pad=20)
    ax2.set_xlim(age_range)
    ax2.set_ylim(age_range)
    ax2.grid(True, alpha=0.3)
    
    # Add correlation and MAE text
    ax2.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Plot 3: Colored by tissue type (bottom-left)
    ax3 = axes[1, 0]
    if 'tissue_type' in merged_data.columns and merged_data['tissue_type'].nunique() > 1:
        for i, tissue in enumerate(merged_data['tissue_type'].unique()):
            if pd.notna(tissue):
                data_subset = merged_data[merged_data['tissue_type'] == tissue]
                ax3.scatter(data_subset['age'], data_subset['predicted_age'], 
                           alpha=0.7, s=50, color=tissue_palette[i % len(tissue_palette)], 
                           label=tissue, edgecolors='white', linewidth=0.5)
        ax3.legend(title='Tissue Type', bbox_to_anchor=(1.05, 1), loc='upper left', 
                  frameon=True, fancybox=True, shadow=True, ncol=1 if merged_data['tissue_type'].nunique() <= 8 else 2)
        title3 = 'Colored by Tissue Type'
    else:
        ax3.scatter(merged_data['age'], merged_data['predicted_age'], 
                   alpha=0.7, s=50, color='#F18F01', edgecolors='white', linewidth=0.5)
        title3 = 'All Samples'
    
    ax3.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Chronological Age (years)', fontweight='bold')
    ax3.set_ylabel('Predicted Age (years)', fontweight='bold')
    ax3.set_title(title3, fontweight='bold', pad=20)
    ax3.set_xlim(age_range)
    ax3.set_ylim(age_range)
    ax3.grid(True, alpha=0.3)
    
    # Add correlation and MAE text
    ax3.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
             transform=ax3.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Plot 4: Plain uncolored (bottom-right)
    ax4 = axes[1, 1]
    ax4.scatter(merged_data['age'], merged_data['predicted_age'], 
               alpha=0.7, s=50, color='#2E86AB', edgecolors='white', linewidth=0.5)
    
    ax4.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2)
    ax4.set_xlabel('Chronological Age (years)', fontweight='bold')
    ax4.set_ylabel('Predicted Age (years)', fontweight='bold')
    ax4.set_title('All Samples', fontweight='bold', pad=20)
    ax4.set_xlim(age_range)
    ax4.set_ylim(age_range)
    ax4.grid(True, alpha=0.3)
    
    # Add correlation and MAE text
    ax4.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the combined plot
    combined_path = output_dir / 'epigenetic_age_predictions_combined.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'epigenetic_age_predictions_combined.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"Combined plot saved to: {combined_path}")
    
    # Create individual high-resolution plots for each panel
    create_individual_plots(merged_data, output_dir, overall_corr, overall_p, overall_mae, age_range)
    
    plt.close()


def create_individual_plots(merged_data: pd.DataFrame, output_dir: Path, 
                          overall_corr: float, overall_p: float, overall_mae: float, age_range: list) -> None:
    """Create individual high-resolution plots for each category."""
    
    # Individual plot for dataset
    if 'dataset' in merged_data.columns and merged_data['dataset'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        dataset_palette = sns.color_palette("Set2", n_colors=merged_data['dataset'].nunique())
        
        for i, dataset in enumerate(merged_data['dataset'].unique()):
            data_subset = merged_data[merged_data['dataset'] == dataset]
            ax.scatter(data_subset['age'], data_subset['predicted_age'], 
                      alpha=0.7, s=60, color=dataset_palette[i], label=dataset, 
                      edgecolors='white', linewidth=0.8)
        
        ax.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2.5, label='Perfect Prediction')
        ax.set_xlabel('Chronological Age (years)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Predicted Age (years)', fontweight='bold', fontsize=14)
        ax.set_title('Epigenetic Age Prediction by Dataset', fontweight='bold', fontsize=16, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(age_range)
        ax.set_ylim(age_range)
        ax.grid(True, alpha=0.3)
        
        ax.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'epigenetic_age_by_dataset.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'epigenetic_age_by_dataset.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Individual plot for gender
    if 'gender' in merged_data.columns and merged_data['gender'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        gender_palette = {'Male': '#1f77b4', 'Female': '#ff7f0e', 'M': '#1f77b4', 'F': '#ff7f0e'}
        
        for gender in merged_data['gender'].unique():
            if pd.notna(gender):
                data_subset = merged_data[merged_data['gender'] == gender]
                color = gender_palette.get(gender, '#2E86AB')
                ax.scatter(data_subset['age'], data_subset['predicted_age'], 
                          alpha=0.7, s=60, color=color, label=gender, 
                          edgecolors='white', linewidth=0.8)
        
        ax.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2.5, label='Perfect Prediction')
        ax.set_xlabel('Chronological Age (years)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Predicted Age (years)', fontweight='bold', fontsize=14)
        ax.set_title('Epigenetic Age Prediction by Gender', fontweight='bold', fontsize=16, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(age_range)
        ax.set_ylim(age_range)
        ax.grid(True, alpha=0.3)
        
        ax.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'epigenetic_age_by_gender.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'epigenetic_age_by_gender.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Individual plot for tissue type
    if 'tissue_type' in merged_data.columns and merged_data['tissue_type'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        tissue_palette = sns.color_palette("tab10", n_colors=merged_data['tissue_type'].nunique())
        
        for i, tissue in enumerate(merged_data['tissue_type'].unique()):
            if pd.notna(tissue):
                data_subset = merged_data[merged_data['tissue_type'] == tissue]
                ax.scatter(data_subset['age'], data_subset['predicted_age'], 
                          alpha=0.7, s=60, color=tissue_palette[i % len(tissue_palette)], 
                          label=tissue, edgecolors='white', linewidth=0.8)
        
        ax.plot(age_range, age_range, '--', color='red', alpha=0.8, linewidth=2.5, label='Perfect Prediction')
        ax.set_xlabel('Chronological Age (years)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Predicted Age (years)', fontweight='bold', fontsize=14)
        ax.set_title('Epigenetic Age Prediction by Tissue Type', fontweight='bold', fontsize=16, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True,
                 ncol=1 if merged_data['tissue_type'].nunique() <= 8 else 2)
        ax.set_xlim(age_range)
        ax.set_ylim(age_range)
        ax.grid(True, alpha=0.3)
        
        ax.text(0.05, 0.80, f'r = {overall_corr:.3f}\np = {overall_p:.2e}\nMAE = {overall_mae:.1f} yrs', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'epigenetic_age_by_tissue.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / 'epigenetic_age_by_tissue.pdf', bbox_inches='tight', facecolor='white')
        plt.close()


def evaluate(predictions_path: Path, eval_data_path: Path, output_dir: Path = None) -> Dict[str, Any]:
    """
    Evaluate predictions against PRIVATE held-out test data only.
    
    This is the FINAL evaluation that uses only the private test set to prevent
    overfitting to the public test set that agents can access during development.
    
    Args:
        predictions_path: Path to predictions file
        eval_data_path: Path to evaluation data directory
        output_dir: Directory to save plots (optional)
        
    Returns:
        Dictionary of evaluation results
    """
    # Load predictions
    predictions = load_predictions(predictions_path)
    
    # Ensure predictions has sample_id as a column
    predictions.columns = predictions.columns.str.lower()
    age_cols_valid = ['age', 'age_years', 'predicted_age']
    for col in age_cols_valid:
        if col in predictions.columns:
            predictions = predictions.rename(columns={col: 'predicted_age'})
            break
    sample_id_cols_valid = ['sample_id', 'sampleid']
    for col in sample_id_cols_valid:
        if col in predictions.columns:
            predictions = predictions.rename(columns={col: 'sample_id'})
            break
    if 'sample_id' not in predictions.columns:
        raise ValueError("Predictions must have a 'sample_id' column")
    
    # Load PRIVATE test metadata only
    private_path = eval_data_path / 'meta_heldout_private.arrow'
    if not private_path.exists():
        raise FileNotFoundError(f"Private evaluation metadata not found: {private_path}")
    
    metadata = pd.read_feather(private_path)
    
    # Convert all column names to lowercase
    metadata.columns = metadata.columns.str.lower()
    
    # Handle sample_id
    sample_id_cols_valid = ['sample_id', 'sampleid']
    for col in sample_id_cols_valid:
        if col in metadata.columns:
            metadata = metadata.rename(columns={col: 'sample_id'})
            break
    
    # Handle age column
    age_cols_valid = ['age', 'age_years']
    for col in age_cols_valid:
        if col in metadata.columns:
            metadata = metadata.rename(columns={col: 'age'})
            break
    
    # Reset index to make sample_id a column if it's in the index
    if 'sample_id' not in metadata.columns:
        metadata = metadata.reset_index()
        # Rename the index column to sample_id if needed
        if 'index' in metadata.columns:
            metadata = metadata.rename(columns={'index': 'sample_id'})
    
    # Merge predictions with ground truth
    merged = metadata.merge(predictions, on='sample_id', how='inner')

    
    if len(merged) == 0:
        raise ValueError("No matching samples found between predictions and ground truth")
    
    if len(merged) < len(metadata):
        print(f"Warning: Only {len(merged)}/{len(metadata)} samples have predictions")
    
    # Create beautiful plots if output directory is provided
    if output_dir:
        print("Creating publication-ready scatter plots...")
        create_publication_plots(merged, output_dir)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(
        merged['age'].values,
        merged['predicted_age'].values
    )
    
    # Calculate metrics by tissue type if available
    tissue_metrics = {}
    if 'tissue_type' in merged.columns:
        for tissue in merged['tissue_type'].unique():
            tissue_data = merged[merged['tissue_type'] == tissue]
            if len(tissue_data) >= 10:  # Only calculate if sufficient samples
                tissue_metrics[tissue] = calculate_metrics(
                    tissue_data['age'].values,
                    tissue_data['predicted_age'].values
            )
    
    # Check if performance criteria are met
    criteria_met = {
        'pearson_correlation >= 0.9': overall_metrics['pearson_correlation'] >= 0.9,
        'mae_years < 10.0': overall_metrics['mae_years'] < 10.0
    }
    
    return {
        'evaluation_type': 'FINAL (Private Test Set Only)',
        'overall_metrics': overall_metrics,
        'tissue_metrics': tissue_metrics,
        'criteria_met': criteria_met,
        'all_criteria_passed': all(criteria_met.values()),
        'evaluation_summary': {
            'total_samples': len(metadata),
            'evaluated_samples': len(merged),
            'missing_predictions': len(metadata) - len(merged)
        }
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Final evaluation of epigenetic clock predictions (PRIVATE test set only)'
    )
    parser.add_argument('--predictions', required=True, help='Path to predictions file')
    parser.add_argument('--eval-data', required=True, help='Path to evaluation data directory')
    
    args = parser.parse_args()
    
    try:
        # Create final_eval_results directory next to predictions file
        predictions_path = Path(args.predictions)
        results_dir = predictions_path.parent / 'final_eval_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set output paths within the results directory
        output_json = results_dir / 'evaluation_results.json'
        plots_dir = results_dir / 'plots'
        
        results = evaluate(
            predictions_path,
            Path(args.eval_data),
            plots_dir
        )
        
        # Save results to the final_eval_results directory
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== FINAL EVALUATION RESULTS (Private Test Set) ===")
        print("This evaluation uses ONLY the private test set to prevent overfitting")
        print("to the public test set that agents can access during development.\n")
        
        overall_metrics = results['overall_metrics']
        print(f"Pearson Correlation: {overall_metrics['pearson_correlation']:.4f}")
        print(f"MAE: {overall_metrics['mae_years']:.2f} years")
        print(f"RMSE: {overall_metrics['rmse_years']:.2f} years")
        print(f"Samples Evaluated: {results['evaluation_summary']['evaluated_samples']}")
        
        if results['all_criteria_passed']:
            print("\n✓ PASSED: All performance criteria met!")
        else:
            print("\n✗ FAILED: Some criteria not met:")
            for criterion, passed in results['criteria_met'].items():
                print(f"  {'✓' if passed else '✗'} {criterion}")
        
        # Print subgroup analysis if available
        if results['tissue_metrics']:
            print("\nPerformance by tissue type:")
            for tissue, metrics in results['tissue_metrics'].items():
                print(f"  {tissue}: Pearson={metrics['pearson_correlation']:.3f}, MAE={metrics['mae_years']:.1f}")
        
        print(f"\n📁 All results saved to: {results_dir}")
        print(f"   📊 Plots: {plots_dir}")
        print(f"   📄 Results JSON: {output_json}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 