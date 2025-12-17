# E-Cell Megathon 25 - Auto Insurance Churn Prediction

ML pipeline for predicting customer churn using clustering-enhanced XGBoost models, meta-model ensembles, and SHAP explainability.

## Key Features
- **Clustering-Enhanced Models**: 4 cluster types (Demographics, Financial, Geographic, Policy)
- **Meta-Model Ensemble**: Logistic regression combining multiple GBM predictions  
- **SHAP Explainability**: Global, regional, and individual customer insights
- **Complete Pipeline**: Training, inference, analysis, and prediction refinement

## Quick Start
```bash
# 1. Generate clustering features
python clustering_analysis.py --input csv_files/autoinsurance_churn.csv --output for_train_full.csv --full

# 2. Train model with clustering
python train.py --input for_train_full.csv --output churn_model.pkl

# 3. Run meta-model pipeline (see META_MODEL_README.md)
python run_multi_inference.py --input data.csv
```

## Documentation
- **[CODEBASE_SUMMARY.md](CODEBASE_SUMMARY.md)** - Complete command reference and technical details
- **[META_MODEL_README.md](META_MODEL_README.md)** - Meta-model pipeline setup and usage
- **[Clustering_Analysis_Report.md](Clustering_Analysis_Report.md)** - Clustering analysis methodology
