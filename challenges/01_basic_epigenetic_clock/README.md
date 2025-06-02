# Basic Epigenetic Clock Challenge

## Overview

The Basic Epigenetic Clock Challenge tasks autonomous agents with developing an epigenetic clock to predict chronological age from DNA methylation data.

## Background

DNA methylation is an epigenetic modification that changes predictably with age across the human lifespan. By measuring methylation levels at specific CpG sites across the genome, researchers can build "epigenetic clocks" that predict chronological age with remarkable accuracy.

## Challenge Description

Agents must develop a machine learning model that:
1. Predicts chronological age from DNA methylation beta values
2. Achieves high accuracy on the held-out test set
3. Provides comprehensive analysis and documentation

## Data Description

Data were obtained from the AltumAge dataset. 

### Training Data (`data/agent/`)

- **`betas.arrow`**: Training DNA methylation beta values
- **`metadata.arrow`**: Training sample metadata
- **`betas_heldout_public.arrow`**: Public test set methylation data (features only)
- **`betas_heldout_private.arrow`**: Private test set methylation data (features only)

### Evaluation Data (`data/eval/`)

- **`meta_heldout_public.arrow`**: Public test set metadata with true ages
- **`meta_heldout_private.arrow`**: Private test set metadata with true ages

## Evaluation Process

### Step 1: Training Phase
Agents use the training data (`betas.arrow` and `metadata.arrow`) to develop an epigenetic clock model.

### Step 2: Prediction Phase
Agents apply their trained model to the held-out test features. If the public evaluation is enabled, agents must generate predictions for the public test set (`betas_heldout_public.arrow`). Once agents are satisfied with their model, they can generate predictions for the private test set (`betas_heldout_private.arrow`).

Agents save their predictions as **`predictions.arrow`** in the working directory with the following structure:
- **Column 1**: `sample_id` - Sample identifiers matching those in `betas_heldout_private.arrow`
- **Column 2**: `predicted_age` - Predicted ages (float type)

### Step 3: Final Evaluation

The evaluation script will:
1. Load the `predictions.arrow` file
2. Match predictions with true ages by `sample_id`
3. Calculate performance metrics (Pearson correlation, MAE, and RMSE)


## Performance Criteria

The challenge requires meeting **ALL** of the following criteria on the private held-out test set:

- **Pearson correlation ≥ 0.9** between predicted and actual age
- **Mean Absolute Error (MAE) < 10 years**

### Required Outputs
1. **`predictions.arrow`**: Predictions for held-out test set 
2. **Model checkpoint**: Trained model that can be loaded for inference
3. **Inference script**: Code to generate predictions on new data
5. **Analysis report**: Markdown report documenting methodology and results

