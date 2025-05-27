# Basic Epigenetic Clock Challenge

## Overview

The Basic Epigenetic Clock Challenge tasks autonomous agents with developing an accurate epigenetic clock to predict chronological age from DNA methylation data. This challenge represents a fundamental problem in aging research and demonstrates the application of machine learning to biological data.

## Background

DNA methylation is an epigenetic modification that changes predictably with age across the human lifespan. By measuring methylation levels at specific CpG sites across the genome, researchers can build "epigenetic clocks" that predict chronological age with remarkable accuracy. These clocks have applications in:

- Aging research and longevity studies
- Forensic age estimation
- Understanding biological vs. chronological aging
- Clinical assessment of aging-related diseases

## Challenge Description

Agents must develop a machine learning model that:
1. Predicts chronological age from DNA methylation beta values
2. Achieves high accuracy on the held-out test set
3. Demonstrates proper train/validation splitting methodology
4. Provides comprehensive analysis and documentation

## Data Description

### Training Data (`data/agent/`)

- **`betas.arrow`**: Training DNA methylation beta values
  - Rows: Samples (n=1,469)
  - Columns: CpG sites (n=485,577)
  - Values: Beta values (0-1, representing methylation levels)
  
- **`metadata.arrow`**: Training sample metadata
  - `sample_id`: Unique sample identifier
  - `age`: Chronological age in years (target variable)
  - `tissue`: Tissue type (blood, brain, etc.)
  - `study_id`: Study of origin
  - Additional covariates

- **`betas_heldout.arrow`**: Held-out test set methylation data (features only)
  - Same format as betas.arrow but for test samples
  - No labels provided - you must generate predictions for these samples

### Evaluation Data (`data/eval/`)

- **`betas_heldout.arrow`**: Copy of test features (same as in agent directory)
- **`meta_heldout.arrow`**: Held-out test set metadata with true ages (NOT accessible to agents)

## Evaluation Process

### Step 1: Training Phase
Use the training data (`betas.arrow` and `metadata.arrow`) to develop your epigenetic clock model.

### Step 2: Prediction Phase
Apply your trained model to the held-out test features (`betas_heldout.arrow`) to generate age predictions.

### Step 3: Output Requirements
Save your predictions as **`predictions.arrow`** in the working directory with the following structure:
- **Column 1**: `sample_id` - Sample identifiers matching those in `betas_heldout.arrow`
- **Column 2**: `predicted_age` - Your predicted ages (float type)

**CRITICAL**: The `predictions.arrow` file is REQUIRED for evaluation. Without it, your solution cannot be scored.

### Step 4: Evaluation
The evaluation script will:
1. Load your `predictions.arrow` file
2. Match predictions with true ages by `sample_id`
3. Calculate performance metrics
4. Generate a detailed evaluation report

### Example Code for Saving Predictions

```python
import pandas as pd

# After generating predictions for the held-out test set
predictions_df = pd.DataFrame({
    'sample_id': test_sample_ids,
    'predicted_age': your_model_predictions
})

# Save in the required format
predictions_df.to_feather('predictions.arrow')
```

See `scripts/prediction_template.py` for a complete example.

## Performance Criteria

The challenge requires meeting **ALL** of the following criteria on the held-out test set:

- **Pearson correlation ≥ 0.9** between predicted and actual age
- **Mean Absolute Error (MAE) < 10 years**

### Required Outputs
1. **`predictions.arrow`**: Predictions for held-out test set (REQUIRED for evaluation)
2. **Model checkpoint**: Trained model that can be loaded for inference
3. **Inference script**: Code to generate predictions on new data
4. **Evaluation results**: Your own internal performance metrics from train/validation splits
5. **Analysis report**: Markdown report documenting methodology and results

## Key Challenges

1. **Batch Effects**: Data comes from multiple studies with potential technical differences
2. **Tissue Heterogeneity**: Different tissues may have distinct aging patterns
3. **Feature Selection**: 485K+ CpG sites require dimensionality reduction
4. **Generalization**: Model must work across diverse populations and conditions
5. **Data Splitting**: Create proper train/validation splits by dataset to avoid data leakage

## Evaluation Script Usage

```bash
# After creating predictions.arrow, run:
python scripts/evaluate.py \
    --predictions predictions.arrow \
    --eval-data data/eval/ \
    --output evaluation_results.json
```

The evaluation script calculates:
- Overall performance metrics
- Performance by tissue type
- Performance by age group (young/middle/old)
- Compliance with all performance criteria

## Expected Approach

Successful solutions typically involve:

1. **Exploratory Data Analysis**
   - Age distribution analysis
   - Tissue type characterization
   - Missing data assessment
   - Batch effect visualization

2. **Data Preprocessing**
   - Quality control and filtering
   - Normalization strategies
   - Batch effect correction
   - Feature selection/dimensionality reduction

3. **Model Development**
   - Algorithm selection (linear regression, elastic net, random forest, neural networks)
   - Hyperparameter optimization
   - Feature importance analysis
   - Model interpretation

4. **Internal Validation Strategy**
   - Train/validation splitting by dataset
   - Performance evaluation on validation set
   - Robustness testing

5. **Final Predictions**
   - Apply model to `betas_heldout.arrow`
   - Save as `predictions.arrow`
   - Verify format before submission

6. **Analysis Report**
   - Executive summary of approach and results
   - Key visualizations with interpretation
   - Performance metrics summary
   - Discussion of strengths and limitations

## Baseline Performance

Historical epigenetic clocks have achieved:
- Horvath Clock (2013): r ≈ 0.96, MAE ≈ 3.6 years
- Hannum Clock (2013): r ≈ 0.96, MAE ≈ 4.9 years
- PhenoAge (2018): r ≈ 0.94, MAE ≈ 5.4 years

The challenge criteria are set to encourage strong performance while being achievable.

## References

1. Horvath, S. (2013). DNA methylation age of human tissues and cell types. Genome Biology, 14(10), R115.
2. Hannum, G., et al. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. Molecular Cell, 49(2), 359-367.
3. Levine, M. E., et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan. Aging, 10(4), 573-591.
4. Bell, C. G., et al. (2019). DNA methylation aging clocks: challenges and recommendations. Genome Biology, 20(1), 249.

## Tips for Success

1. **Generate Predictions**: Don't forget to create `predictions.arrow` for the held-out test set
2. **Verify Format**: Ensure your predictions file has the correct columns and data types
3. **Start Simple**: Begin with basic linear models before moving to complex architectures
4. **Validate Internally**: Use proper train/validation splits from the training data
5. **Document Everything**: Maintain detailed records in the lab notebook
6. **Consider Biology**: Leverage domain knowledge about aging and methylation
7. **Check Data Leakage**: Ensure datasets don't appear in both train and validation splits
8. **Iterate Systematically**: Build complexity gradually with proper validation
9. **Write Clear Report**: Document your methodology and results in a concise markdown report

## Support

For questions about this challenge:
- Check the `scripts/prediction_template.py` for prediction format examples
- Review the `task.yaml` for detailed specifications
- Examine the evaluation script for metric calculations
- Consult the BioAgents framework documentation


