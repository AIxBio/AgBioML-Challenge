"""Public test set evaluation for biomlagents."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy.stats import pearsonr


class PublicEvaluator:
    """
    Handles evaluation on the PUBLIC test subset only.
    
    Agents can evaluate on public subset multiple times to refine their models.
    This class has NO ACCESS to private test labels - those are handled separately
    by the pipeline after agents complete their work.
    """
    
    def __init__(self, public_labels_path: Path):
        """
        Initialize evaluator with PUBLIC labels only.
        
        Args:
            public_labels_path: Path to meta_heldout_public.arrow containing public test labels
        """
        if not public_labels_path.exists():
            raise FileNotFoundError(f"Public labels not found: {public_labels_path}")
            
        # Load ONLY public labels - this is all we need
        self.public_labels = pd.read_feather(public_labels_path)
        
        # Extract public sample IDs and ages
        self.public_sample_ids = self.public_labels.index.tolist()
        self.public_ages = self.public_labels['age'].values
        
        # Track attempts
        self.public_attempts = 0
        self.max_attempts = 1000
        
    def evaluate(self, predictions_path: str) -> Dict[str, Any]:
        """
        Evaluate predictions on the public test subset.
        
        Args:
            predictions_path: Path to predictions file
            
        Returns:
            Dictionary with evaluation results
        """
        if self.public_attempts >= self.max_attempts:
            return {
                "status": "error",
                "message": f"Maximum attempts ({self.max_attempts}) reached"
            }
        
        try:
            # Load predictions
            predictions_df = pd.read_feather(predictions_path)
            
            # Validate format
            if 'sample_id' not in predictions_df.columns or 'predicted_age' not in predictions_df.columns:
                return {
                    "status": "error",
                    "message": "Predictions must have 'sample_id' and 'predicted_age' columns"
                }
            
            # Match predictions with public labels
            predictions_df = predictions_df.set_index('sample_id')
            
            # Get predictions for public samples in the correct order
            try:
                predicted_ages = predictions_df.loc[self.public_sample_ids, 'predicted_age'].values
            except KeyError as e:
                missing_ids = set(self.public_sample_ids) - set(predictions_df.index)
                return {
                    "status": "error",
                    "message": f"Missing predictions for {len(missing_ids)} public samples"
                }
            
            # Calculate metrics
            pearson_r, p_value = pearsonr(self.public_ages, predicted_ages)
            mae = np.mean(np.abs(self.public_ages - predicted_ages))
            rmse = np.sqrt(np.mean((self.public_ages - predicted_ages) ** 2))
            
            self.public_attempts += 1
            
            return {
                "status": "success",
                "pearson_correlation": float(pearson_r),
                "p_value": float(p_value),
                "mae_years": float(mae),
                "rmse_years": float(rmse),
                "n_samples": len(self.public_sample_ids),
                "attempt": self.public_attempts,
                "max_attempts": self.max_attempts
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error evaluating predictions: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current evaluation status."""
        return {
            "public_attempts": self.public_attempts,
            "max_attempts": self.max_attempts,
            "attempts_remaining": self.max_attempts - self.public_attempts,
            "n_samples": len(self.public_sample_ids)
        }
    
    def get_public_attempts(self) -> int:
        """Get the number of public evaluation attempts used."""
        return self.public_attempts
    
    def get_sample_ids(self) -> list:
        """Get the list of public sample IDs for validation."""
        return self.public_sample_ids.copy()

