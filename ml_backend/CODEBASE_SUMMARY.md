# E-Cell Megathon 25 - Churn Prediction Codebase

## 🎯 Overview
Complete ML pipeline for auto insurance churn prediction featuring:
- **Clustering-Enhanced XGBoost Models** with 4 cluster types (Demographics, Financial, Geographic, Policy)
- **SHAP Explainability** for global, local, and individual customer insights
- **Meta-Model Ensemble** using logistic regression to combine multiple GBM predictions
- **Individual Customer Analysis** tools for specific customer explanations

## 📁 Project Structure

### Core Training & Inference
- `train.py` - Advanced XGBoost training with clustering features
- `x.py` - Inference engine for trained models
- `model.py` - Basic XGBoost training (no clustering)

### Meta-Model Pipeline
- `run_multi_inference.py` - Runs inference across multiple models
- `prepare_metamodel_dataset.py` - Combines model outputs for ensemble
- `train_metamodel.py` - Trains/uses logistic regression meta-model

### Clustering & Analysis
- `clustering_analysis.py` - K-means clustering with bias prevention
- `explain.py` - Global/regional SHAP analysis
- `explain_individual.py` - Individual customer SHAP explanations
- `refine_predictions.py` - Post-inference prediction optimization

### Data & Models
- `csv_files/` - Training and inference datasets
- `multi_model_outputs/` - Individual model predictions
- `metamodel_artifacts/` - Trained meta-model components
- `final_churn_predictions.csv` - Ultimate ensemble predictions

## 🚀 Quick Start Commands

### 1. Data Preparation & Clustering
```bash
# Generate clustering features
python clustering_analysis.py --input csv_files/autoinsurance_churn.csv --output for_train_full.csv --full
```

### 2. Train Multiple Models
```bash
# Train cluster-enhanced models
python train.py --input for_train_37.5k.csv --model-out churn_model_37.5k_cluster_v2.pkl --features-out model_features_37.5k_cluster_v2.pkl --metadata-out model_metadata_37.5k_cluster_v2.pkl

python train.py --input for_train_37.5k.csv --model-out churn_model_37.5k_cluster_v3.pkl --features-out model_features_37.5k_cluster_v3.pkl --metadata-out model_metadata_37.5k_cluster_v3.pkl

python train.py --input for_train_full.csv --model-out churn_model_full_cluster.pkl --features-out model_features_full_cluster.pkl --metadata-out model_metadata_full_cluster.pkl

# Or basic XGBoost (edit CSV_FILE_PATH in script)
python model.py
```

### 3. Run Meta-Model Pipeline
```bash
# Step 1: Multi-model inference
python run_multi_inference.py --input for_inf_after37.5_v2.csv --output-dir multi_model_outputs --python "C:\Users\Lenovo\OneDrive\Desktop\megathon\E-Cell-Megathon-25\.venv\Scripts\python.exe"

# Step 2: Prepare ensemble dataset
python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv --add-ensemble

# Step 3: Train meta-model
python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts

# Step 4: Final predictions
python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_churn_predictions.csv
```

### 4. Explainability & Analysis
```bash
# Global SHAP analysis
python explain.py --data-csv for_train_full.csv --model-pkl churn_model_full_cluster.pkl --features-pkl model_features_full_cluster.pkl --metadata-json model_metadata_full_cluster.json

# Individual customer analysis
python explain_individual.py --csv inf_after37.5_v3.csv --model churn_model_37.5k_cluster.pkl --features model_features_37.5k_cluster.pkl --metadata model_metadata_37.5k_cluster.json --individual-id "221300574354"
```

## 📖 Detailed Command Reference

### train.py - Advanced Model Training
```bash
python train.py [OPTIONS]

Required:
  --input INPUT_CSV          # Training data with clustering features
  --model-out MODEL.pkl      # Output model file
  --features-out FEATURES.pkl # Output feature list
  --metadata-out META.json   # Output metadata for SHAP

Optional:
  --target TARGET            # Target column (default: Churn)
  --drop-cols COLS           # Comma-separated columns to drop
  --test-size FLOAT          # Test split ratio (default: 0.25)
  --n-iter INT               # RandomizedSearchCV iterations (default: 20)
  --cv-splits INT            # Cross-validation folds (default: 5)
  --n-jobs INT               # Parallel jobs (default: -1)
  --random-state INT         # Random seed (default: 42)

Examples:
  python train.py --input for_train_full.csv --model-out my_model.pkl --features-out my_features.pkl --metadata-out my_metadata.json --n-iter 50 --cv-splits 10
```

### x.py - Inference Engine
```bash
python x.py [OPTIONS]

Required:
  --input INPUT_CSV          # Input data for inference
  --model MODEL.pkl          # Trained model file
  --features FEATURES.pkl    # Feature list file
  --metadata METADATA.json   # Model metadata file

Optional:
  --output OUTPUT_CSV        # Output file (default: INPUT_inference.csv)
  --target TARGET            # Ground truth column if available
  --drop-cols COLS           # Columns to drop
  --threshold FLOAT          # Decision threshold (default: 0.5)

Examples:
  python x.py --input new_data.csv --output predictions.csv --model churn_model.pkl --features model_features.pkl --metadata model_metadata.json
  python x.py --input test_data.csv --model my_model.pkl --features my_features.pkl --metadata my_metadata.json --threshold 0.3
```

### run_multi_inference.py - Multi-Model Pipeline
```bash
python run_multi_inference.py [OPTIONS]

Required:
  --input INPUT_CSV          # Input data for all models

Optional:
  --output-dir DIR           # Output directory (default: multi_model_outputs)
  --python PYTHON_PATH       # Python executable path

Target Models (auto-detected):
  - churn_model_37.5k_cluster_v2.pkl
  - churn_model_37.5k_cluster_v3.pkl
  - churn_model_37.5k_cluster.pkl
  - churn_model_full_cluster.pkl

Examples:
  python run_multi_inference.py --input data.csv --output-dir my_outputs
  python run_multi_inference.py --input data.csv --python "/path/to/python"
```

### prepare_metamodel_dataset.py - Dataset Preparation
```bash
python prepare_metamodel_dataset.py [OPTIONS]

Optional:
  --input-dir DIR            # Directory with model outputs (default: multi_model_outputs)
  --output FILE              # Output dataset (default: metamodel_dataset.csv)
  --add-ensemble             # Add ensemble features (mean, std, etc.)

Examples:
  python prepare_metamodel_dataset.py --input-dir outputs --output ensemble_data.csv --add-ensemble
```

### train_metamodel.py - Meta-Model Training/Prediction
```bash
# Training Mode
python train_metamodel.py --mode train --input metamodel_dataset.csv [OPTIONS]

Optional for Training:
  --output-dir DIR           # Output directory (default: metamodel_outputs)

# Prediction Mode  
python train_metamodel.py --mode predict --input DATA.csv --model MODEL.pkl --scaler SCALER.pkl --features FEATURES.txt --output-file OUTPUT.csv

Examples:
  # Train
  python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir my_metamodel
  
  # Predict
  python train_metamodel.py --mode predict --input new_data.csv --model metamodel.pkl --scaler scaler.pkl --features features.txt --output-file final_predictions.csv
```

### clustering_analysis.py - Clustering Features
```bash
python clustering_analysis.py --input INPUT.csv --output OUTPUT.csv [OPTIONS]

Optional:
  --full                     # Use full dataset (no sampling)
  --sample-size INT          # Sample size if not using --full

Examples:
  python clustering_analysis.py --input autoinsurance_churn.csv --output clustered_data.csv --full
  python clustering_analysis.py --input data.csv --output clustered_data.csv --sample-size 50000
```

### explain.py - SHAP Analysis
```bash
python explain.py --data-csv DATA.csv --model-pkl MODEL.pkl --features-pkl FEATURES.pkl --metadata-json METADATA.json [OPTIONS]

Optional:
  --output-dir DIR           # Output directory
  --n-samples INT            # Sample size for SHAP

Examples:
  python explain.py --data-csv train_data.csv --model-pkl model.pkl --features-pkl features.pkl --metadata-json metadata.json --n-samples 1000
```

### explain_individual.py - Individual Analysis
```bash
python explain_individual.py --csv DATA.csv --model MODEL.pkl --features FEATURES.pkl --metadata METADATA.json --individual-id ID [OPTIONS]

Required:
  --individual-id ID         # Specific customer ID to analyze

Examples:
  python explain_individual.py --csv customer_data.csv --model model.pkl --features features.pkl --metadata metadata.json --individual-id "12345"
```

## 🎯 Key Outputs

1. **Model Files**: `churn_model_*.pkl` - Trained XGBoost models
2. **Ensemble Predictions**: `final_churn_predictions.csv` - Meta-model results
3. **SHAP Reports**: `shap_reports/` - Explainability analysis
4. **Individual Explanations**: `individual_explanations/` - Customer-specific insights
5. **Performance Metrics**: Embedded in model metadata and console output

## 📊 Performance Summary

- **Individual Models**: AUC ~0.85-0.87 on test sets
- **Meta-Model**: AUC ~0.69 (ensemble of 4 models)
- **Dataset**: 1.6M+ customers, 20+ features including clustering
- **Churn Rate**: ~11.5% (realistic imbalanced dataset)

## 🔧 Technical Notes

- **Python Environment**: Requires pandas, scikit-learn, xgboost, shap, joblib
- **Memory**: Large datasets require sufficient RAM (8GB+ recommended)
- **Clustering**: Bias prevention ensures no target leakage
- **SHAP**: TreeExplainer with grouped feature analysis
- **Meta-Model**: Logistic regression with StandardScaler and cross-validation