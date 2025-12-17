"""
SHAP Analysis Module for Customer Churn Prediction Dashboard

This module accepts JSON input containing customer information and returns
SHAP analysis results for visualization on the dashboard.

Usage:
    python shap_analysis.py --input customer_data.json
    
Or import as a module:
    from shap_analysis import analyze_customer_churn
"""

import pandas as pd
import numpy as np
import json
import sys
import argparse
from pathlib import Path

# Import shap with explicit reference to avoid conflicts
import shap as shap_lib
import joblib

# --- Configuration ---
MODEL_FILENAME = 'churn_model.pkl'

CATEGORICAL_COLS = [
    'city',
    'marital_status',
    'acct_suspd_date',
    'cust_orig_date',
    'state',
    'county',
    'home_market_value'
]

IDENTIFIER_COLS = ['individual_id', 'address_id']


class CustomerChurnAnalyzer:
    """
    Analyzes customer churn probability using SHAP values.
    Accepts customer data in JSON format and returns dashboard-ready results.
    """
    
    def __init__(self, model_path=MODEL_FILENAME):
        """Initialize the analyzer with model."""
        print(f"🔄 Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # Initialize SHAP explainer
        print("🔄 Initializing SHAP explainer...")
        self.explainer = shap_lib.TreeExplainer(self.model)
        
        # Store feature names from model
        self.model_features = self.model.get_booster().feature_names
        
        print("✅ CustomerChurnAnalyzer initialized successfully!")
    
    def preprocess_customer_data(self, customer_dict):
        """
        Preprocess customer data to match model requirements.
        
        Args:
            customer_dict: Dictionary containing customer information
            
        Returns:
            DataFrame aligned with model features
        """
        # Create DataFrame from customer dict
        df = pd.DataFrame([customer_dict])
        
        # Remove identifier columns if present
        df_features = df.drop(columns=[col for col in IDENTIFIER_COLS if col in df.columns], errors='ignore')
        
        # One-hot encode categorical columns
        categorical_present = [col for col in CATEGORICAL_COLS if col in df_features.columns]
        df_encoded = pd.get_dummies(df_features, columns=categorical_present)
        
        # Align with model features (add missing columns, remove extra ones)
        df_aligned = df_encoded.reindex(columns=self.model_features, fill_value=0)
        
        return df_aligned
    
    def analyze_customer(self, customer_dict):
        """
        Analyze a customer and return SHAP values with prediction.
        
        Args:
            customer_dict: Dictionary with complete customer data
                
        Returns:
            Dictionary with prediction, SHAP values, and feature importance
        """
        if not isinstance(customer_dict, dict):
            return {
                "error": "Customer data must be a dictionary",
                "success": False
            }
        
        customer_id = customer_dict.get('individual_id', 'Unknown')
        
        # Preprocess the data
        customer_features = self.preprocess_customer_data(customer_dict)
        
        # Get prediction
        prediction_proba = self.model.predict_proba(customer_features)[0]
        churn_probability = float(prediction_proba[1])
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(customer_features)
        
        # Format SHAP values for dashboard
        shap_data = []
        for feature_name, shap_value in zip(customer_features.columns, shap_values[0]):
            # Extract original feature name (before one-hot encoding)
            original_feature = feature_name
            for cat_col in CATEGORICAL_COLS:
                if feature_name.startswith(cat_col):
                    original_feature = cat_col
                    break
            
            shap_data.append({
                "feature": feature_name,
                "original_feature": original_feature,
                "shap_value": float(shap_value),
                "feature_value": float(customer_features[feature_name].iloc[0]),
                "impact": "increases_churn" if shap_value > 0 else "decreases_churn"
            })
        
        # Sort by absolute SHAP value
        shap_data_sorted = sorted(shap_data, key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Aggregate by original feature for cleaner dashboard display
        aggregated_shap = {}
        for item in shap_data:
            orig_feat = item['original_feature']
            if orig_feat not in aggregated_shap:
                aggregated_shap[orig_feat] = {
                    "feature": orig_feat,
                    "total_shap_value": 0,
                    "impact": item['impact']
                }
            aggregated_shap[orig_feat]['total_shap_value'] += item['shap_value']
        
        aggregated_list = sorted(
            aggregated_shap.values(),
            key=lambda x: abs(x['total_shap_value']),
            reverse=True
        )
        
        # Return comprehensive analysis
        result = {
            "success": True,
            "customer_id": str(customer_id),
            "prediction": {
                "churn_probability": churn_probability,
                "will_churn": churn_probability > 0.5,
                "confidence": float(max(prediction_proba[0], prediction_proba[1]))
            },
            "shap_analysis": {
                "base_value": float(self.explainer.expected_value),
                "top_features": shap_data_sorted[:15],  # Top 15 individual features
                "aggregated_features": aggregated_list[:10],  # Top 10 aggregated features
                "total_features_analyzed": len(shap_data)
            },
            "interpretation": self._generate_interpretation(shap_data_sorted[:5], churn_probability)
        }
        
        return result
    
    def _generate_interpretation(self, top_features, churn_prob):
        """Generate human-readable interpretation of the analysis."""
        interpretation = {
            "risk_level": "high" if churn_prob > 0.7 else "medium" if churn_prob > 0.4 else "low",
            "key_factors": []
        }
        
        for feat in top_features:
            direction = "increasing" if feat['shap_value'] > 0 else "decreasing"
            interpretation['key_factors'].append({
                "feature": feat['original_feature'],
                "impact": f"{direction} churn risk",
                "magnitude": abs(feat['shap_value'])
            })
        
        return interpretation
    
    def batch_analyze(self, customer_data_list):
        """
        Analyze multiple customers at once.
        
        Args:
            customer_data_list: List of customer data dictionaries
            
        Returns:
            Dictionary with batch results
        """
        results = []
        for customer_data in customer_data_list:
            result = self.analyze_customer(customer_data)
            results.append(result)
        
        return {
            "batch_results": results,
            "total_analyzed": len(results),
            "successful": sum(1 for r in results if r.get('success', False))
        }


def analyze_customer_churn(customer_dict, model_path=MODEL_FILENAME):
    """
    Convenience function to analyze a single customer.
    
    Args:
        customer_dict: Dictionary with customer data
        model_path: Path to the trained model file
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = CustomerChurnAnalyzer(model_path)
    return analyzer.analyze_customer(customer_dict)


def main():
    """Command-line interface for the analyzer."""
    parser = argparse.ArgumentParser(description='Analyze customer churn using SHAP')
    parser.add_argument('--input', type=str, required=True, help='Path to JSON file with customer data')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--model', type=str, default=MODEL_FILENAME, help='Path to model file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CustomerChurnAnalyzer(args.model)
    
    # Load customer data from JSON
    with open(args.input, 'r') as f:
        customer_data = json.load(f)
    
    # Run analysis
    print(f"\n🔍 Analyzing customer...")
    result = analyzer.analyze_customer(customer_data)
    
    # Output results
    result_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result_json)
        print(f"✅ Results saved to {args.output}")
    else:
        print("\n📊 Analysis Results:")
        print(result_json)


if __name__ == "__main__":
    main()
