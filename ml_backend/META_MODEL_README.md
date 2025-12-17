# Meta-Model Pipeline for Churn Prediction

This pipeline creates a logistic regression meta-model that combines predictions from multiple GBM models to produce a final churn probability.

## Overview

The pipeline consists of 4 steps:
1. **Multi-Model Inference**: Run inference across all trained GBM models
2. **Dataset Preparation**: Combine model outputs into a training dataset
3. **Meta-Model Training**: Train logistic regression on combined predictions
4. **Final Predictions**: Use meta-model to generate final churn probabilities

## Files Created

### Core Scripts
- `model.py` - Basic XGBoost training (creates models without clustering features)
- `train.py` - Advanced training with clustering features (RECOMMENDED for meta-model)
- `run_multi_inference.py` - Runs x.py across all models
- `prepare_metamodel_dataset.py` - Combines model outputs 
- `train_metamodel.py` - Trains and uses logistic regression meta-model
- `run_metamodel_pipeline.ps1` - PowerShell script showing complete pipeline

### Input Requirements
Your workspace should have these model files (created by Step 0 above):
- `churn_model_37.5k_cluster_v2.pkl` + `model_features_37.5k_cluster_v2.pkl` + `model_metadata_37.5k_cluster_v2.pkl`
- `churn_model_37.5k_cluster_v3.pkl` + `model_features_37.5k_cluster_v3.pkl` + `model_metadata_37.5k_cluster_v3.pkl`
- `churn_model_37.5k_cluster.pkl` + `model_features_37.5k_cluster.pkl` + `model_metadata_37.5k_cluster.pkl`
- `churn_model_full_cluster.pkl` + `model_features_full_cluster.pkl` + `model_metadata_full_cluster.pkl`

**Note**: The basic `model.py` creates models without clustering features. For best results, use `train.py` with clustering analysis to create the cluster-enhanced models listed above.

## Usage

### Step 0: Train Individual Models (if needed)

If you need to train new models, use `model.py` for basic XGBoost models or `train.py` for cluster-enhanced models:

```bash
# Basic XGBoost model (using model.py)
python model.py
# Creates: churn_model_50k_no_cluster.pkl, model_features_50k_no_cluster.pkl, shap_explainer_50k_no_cluster.pkl

# Cluster-enhanced models (using train.py) - RECOMMENDED
python train.py --input for_train_37.5k.csv --model-out churn_model_37.5k_cluster_v2.pkl --features-out model_features_37.5k_cluster_v2.pkl --metadata-out model_metadata_37.5k_cluster_v2.pkl

python train.py --input for_train_37.5k.csv --model-out churn_model_37.5k_cluster_v3.pkl --features-out model_features_37.5k_cluster_v3.pkl --metadata-out model_metadata_37.5k_cluster_v3.pkl

python train.py --input for_train_full.csv --model-out churn_model_full_cluster.pkl --features-out model_features_full_cluster.pkl --metadata-out model_metadata_full_cluster.pkl
```

### Step 1: Run Multi-Model Inference

```bash
python run_multi_inference.py --input for_inf_after37.5csv --output-dir multi_model_outputs
```

This will create:
- `multi_model_outputs/inference_37.5k_v2.csv`
- `multi_model_outputs/inference_37.5k_v3.csv`
- `multi_model_outputs/inference_37.5k.csv`  
- `multi_model_outputs/inference_full.csv`

Each CSV has:
- All original columns from input
- Last column: churn probability from that model
- 3rd last column: actual churn (0/1)

### Step 2: Prepare Meta-Model Dataset

```bash
python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv --add-ensemble
```

This creates `metamodel_dataset.csv` with:
- `individual_id` (if available)
- `actual_churn` - ground truth labels
- `prob_37.5k_v2`, `prob_37.5k_v3`, etc. - individual model predictions
- Ensemble features: `prob_mean`, `prob_std`, `prob_min`, `prob_max`, etc.
- Difference features: `diff_v2_v3`, etc. for model agreement

### Step 3: Train Meta-Model

```bash
python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts
```

This creates:
- `metamodel_artifacts/metamodel.pkl` - trained logistic regression
- `metamodel_artifacts/metamodel_scaler.pkl` - feature scaler
- `metamodel_artifacts/metamodel_features.txt` - feature names
- `metamodel_artifacts/metamodel_metadata.json` - performance metrics
- `metamodel_artifacts/metamodel_performance.png` - ROC/PR curves
- `metamodel_artifacts/metamodel_feature_importance.png` - feature importance

### Step 4: Generate Final Predictions

```bash
python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_churn_predictions.csv
```

This creates `final_churn_predictions.csv` with all original columns plus:
- `final_churn_probability` - meta-model output

## Key Features

### Basic Model Training (`model.py`)
- **Purpose**: Creates baseline XGBoost models without clustering features
- **Input**: Configurable CSV file (default: `csv_files/autoinsurance_churn_50000.csv`)
- **Features**:
  - Automated hyperparameter tuning with RandomizedSearchCV
  - Built-in SHAP explainability setup
  - Handles class imbalance with `scale_pos_weight`
  - Creates basic feature engineering with `pd.get_dummies()`
- **Outputs**:
  - `churn_model_50k_no_cluster.pkl` - trained XGBoost model
  - `model_features_50k_no_cluster.pkl` - feature list for inference alignment
  - `shap_explainer_50k_no_cluster.pkl` - SHAP explainer for interpretability
- **Configuration**: Edit CSV_FILE_PATH, TARGET_COLUMN, FEATURES_TO_DROP in script header
- **Usage**: Simply run `python model.py` (no command line arguments)

### Advanced Model Training (`train.py`)
- **Purpose**: Creates cluster-enhanced XGBoost models (RECOMMENDED for meta-model)
- **Features**: Includes clustering features + advanced preprocessing + metadata for SHAP
- **Usage**: Command-line interface with flexible parameters

### Multi-Model Inference (`run_multi_inference.py`)
- Automatically detects available model files
- Handles missing models gracefully
- Provides detailed execution summary
- Validates output files are created

### Dataset Preparation (`prepare_metamodel_dataset.py`)
- Aligns datasets by `individual_id` or row order
- Handles missing ID columns
- Creates ensemble features (mean, std, min, max, etc.)
- Adds model agreement features (pairwise differences)
- Provides comprehensive dataset statistics

### Meta-Model Training (`train_metamodel.py`)
- **Training Mode**:
  - Cross-validation with multiple regularization strengths
  - Automatic model selection based on CV AUC
  - Comprehensive evaluation metrics (AUC, AP, classification report)
  - Feature importance analysis
  - Performance visualizations
- **Prediction Mode**:
  - Loads trained model artifacts
  - Applies same preprocessing pipeline
  - Generates final churn probabilities

## Pipeline Flexibility

### Customization Options
- `--add-ensemble`: Add statistical ensemble features
- `--output-dir`: Specify output directory
- Different test/train splits in meta-model training
- Various regularization strengths tested automatically

### Error Handling
- Missing model files are skipped with warnings
- Dataset alignment handles different ID formats
- Robust feature extraction with fallbacks
- Detailed error messages and execution summaries

## Expected Performance

The meta-model typically achieves:
- Better AUC than individual models
- More stable predictions (lower variance)
- Better calibrated probabilities
- Improved performance on edge cases where models disagree

## Output Format

Final predictions in `final_churn_predictions.csv`:
```
individual_id,feature1,feature2,...,actual_churn,prob_v2,prob_v3,...,final_churn_probability
12345,value1,value2,...,0,0.23,0.31,...,0.27
67890,value1,value2,...,1,0.78,0.82,...,0.80
```

The `final_churn_probability` is the meta-model's prediction and should be used as the final churn score.

## Troubleshooting

### Common Issues
1. **Missing model files**: Ensure all 4 model variants exist with matching features/metadata files
2. **Column alignment**: Verify all model outputs have consistent column structure  
3. **Memory issues**: Use smaller batches if processing very large datasets
4. **Feature mismatch**: Ensure all models were trained with consistent feature sets

### Debugging
- Use `--verbose` flags where available
- Check intermediate CSV files for proper structure
- Review execution summaries for skipped models
- Examine feature importance to validate meta-model learning

## Configuring model.py

To customize the basic XGBoost training in `model.py`, edit these configuration variables at the top of the script:

```python
# --- 1. Configuration ---
CSV_FILE_PATH = 'csv_files/autoinsurance_churn_50000.csv'  # Your input CSV
TARGET_COLUMN = 'Churn'                                    # Target column name
FEATURES_TO_DROP = ['individual_id', 'address_id']         # Columns to exclude
```

### model.py Hyperparameter Grid
The script searches these parameter ranges:
- `n_estimators`: [100, 200, 300, 400]
- `learning_rate`: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
- `max_depth`: [3, 4, 5, 6]
- `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0]
- `colsample_bytree`: [0.6, 0.7, 0.8, 0.9, 1.0]
- `gamma`: [0, 0.1, 0.2, 0.3, 0.4]
- `reg_alpha`: [0, 0.1, 0.5, 1]
- `reg_lambda`: [0.1, 0.5, 1, 1.5, 2]

To modify the search space, edit the `param_dist` dictionary in the script.

### model.py vs train.py Comparison

| Feature | model.py | train.py |
|---------|----------|----------|
| **Command Line** | No arguments | Full CLI interface |
| **Clustering Features** | ❌ No | ✅ Yes (Demographics, Financial, Geographic, Policy) |
| **SHAP Integration** | Basic setup | Full integration with metadata |
| **Hyperparameter Tuning** | RandomizedSearchCV | RandomizedSearchCV + advanced options |
| **Feature Engineering** | Basic `pd.get_dummies()` | Advanced preprocessing pipeline |
| **Model Metadata** | Basic | Comprehensive (for SHAP + meta-model) |
| **Use Case** | Quick baseline models | Production-ready cluster-enhanced models |
| **Recommended For** | Prototyping, simple datasets | Meta-model pipeline, complex datasets |

Run `python run_metamodel_pipeline.ps1` to see the complete pipeline commands.

## Complete Workflow Examples

### Option A: Using Basic Models (model.py)

```bash
# 1. Train basic models (modify CSV_FILE_PATH in each run)
python model.py  # Creates churn_model_50k_no_cluster.pkl

# 2. Update run_multi_inference.py to use basic models
# Edit the models list to point to your basic model files

# 3. Run meta-model pipeline
python run_multi_inference.py --input your_inference_data.csv --output-dir multi_model_outputs
python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv
python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts
python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_predictions.csv
```

### Option B: Using Cluster-Enhanced Models (train.py) - RECOMMENDED

```bash
# 1. Run clustering analysis first
python clustering_analysis.py --input autoinsurance_churn.csv --output for_train_full.csv --full

# 2. Train cluster-enhanced models
python train.py --input for_train_37.5k.csv --model-out churn_model_37.5k_cluster_v2.pkl --features-out model_features_37.5k_cluster_v2.pkl --metadata-out model_metadata_37.5k_cluster_v2.pkl

python train.py --input for_train_37.5k.csv --model-out churn_model_37.5k_cluster_v3.pkl --features-out model_features_37.5k_cluster_v3.pkl --metadata-out model_metadata_37.5k_cluster_v3.pkl

python train.py --input for_train_full.csv --model-out churn_model_full_cluster.pkl --features-out model_features_full_cluster.pkl --metadata-out model_metadata_full_cluster.pkl

# 3. Run meta-model pipeline (as shown in main usage section above)
python run_multi_inference.py --input for_inf_after37.5csv --output-dir multi_model_outputs
# ... rest of pipeline
```