"""Evaluation tools for autobioml framework."""

from typing import Dict, Any, Optional
from pathlib import Path
from autogen_core.tools import FunctionTool

# Global evaluator instance (will be set by pipeline)
_evaluator = None
_public_attempts = 0
MAX_PUBLIC_ATTEMPTS = 1000


async def evaluate_on_public_test(predictions_file: str) -> Dict[str, Any]:
    """
    Evaluate your model predictions on the PUBLIC subset (50%) of the test set.
    
    This allows you to test your approach before final submission. You can use this tool
    multiple times to refine your model based on the feedback.
    
    IMPORTANT: This tool is for PUBLIC test set evaluation only. Your final predictions.arrow
    should contain predictions for the PRIVATE test set (betas_heldout_private.arrow).
    
    Args:
        predictions_file: Path to predictions file (arrow format) with columns:
                         - sample_id: matching IDs from betas_heldout_public.arrow
                         - predicted_age: your model's predictions
                         
                         This file should be generated using betas_heldout_public.arrow
        
    Returns:
        Dictionary with evaluation metrics on the public test set including:
        - pearson_correlation: Correlation between true and predicted ages
        - mae_years: Mean absolute error in years
        - rmse_years: Root mean squared error in years
        - attempts_remaining: Number of evaluation attempts left
    """
    global _evaluator, _public_attempts
    
    if _evaluator is None:
        return {
            "status": "error",
            "message": "Evaluation system not available. Public evaluation may be disabled for this run."
        }
    
    if _public_attempts >= MAX_PUBLIC_ATTEMPTS:
        return {
            "status": "error", 
            "message": f"Maximum public evaluation attempts ({MAX_PUBLIC_ATTEMPTS}) reached. Please finalize your model."
        }
    
    # Convert to Path and handle relative paths
    pred_path = Path(predictions_file)
    if not pred_path.is_absolute():
        pred_path = Path.cwd() / pred_path
    
    if not pred_path.exists():
        return {
            "status": "error",
            "message": f"Predictions file not found: {predictions_file}"
        }
    
    try:
        # Call the evaluator's evaluate method
        results = _evaluator.evaluate(pred_path)
        
        if results["status"] == "success":
            _public_attempts += 1
            results["attempts_remaining"] = MAX_PUBLIC_ATTEMPTS - _public_attempts
            results["message"] = (
                f"Public evaluation {_public_attempts}/{MAX_PUBLIC_ATTEMPTS} complete. "
                f"Pearson r={results['pearson_correlation']:.4f}, MAE={results['mae_years']:.2f} years. "
                f"You have {results['attempts_remaining']} attempts remaining."
            )
        
        return results
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during evaluation: {str(e)}"
        }


def set_evaluator(evaluator) -> None:
    """Set the global evaluator instance."""
    global _evaluator, _public_attempts
    _evaluator = evaluator
    _public_attempts = 0  # Reset attempts when setting new evaluator


def get_evaluation_tools():
    """Get evaluation tool instances."""
    return {
        "evaluate_on_public_test": FunctionTool(
            evaluate_on_public_test,
            name="evaluate_on_public_test",
            description="Evaluate model predictions on the public subset of the test set (50% of data)"
        )
    } 