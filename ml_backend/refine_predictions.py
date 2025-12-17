#!/usr/bin/env python3
"""
Prediction Refinement Script
Applies business rules and threshold optimization to improve churn predictions
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find optimal threshold based on specified metric"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]
    elif metric == 'precision':
        # Find threshold for precision >= 0.8
        high_prec_idx = np.where(precision >= 0.8)[0]
        if len(high_prec_idx) > 0:
            return thresholds[high_prec_idx[0]], precision[high_prec_idx[0]]
    elif metric == 'recall':
        # Find threshold for recall >= 0.9
        high_recall_idx = np.where(recall >= 0.9)[0]
        if len(high_recall_idx) > 0:
            return thresholds[high_recall_idx[-1]], recall[high_recall_idx[-1]]
    
    return 0.5, 0.0

def apply_business_rules(df):
    """Apply business logic refinements"""
    refined_prob = df['predicted_churn_probability'].copy()
    
    # Rule 1: Boost risk for short tenure + poor credit
    short_tenure = df['days_tenure'] < 365
    poor_credit = df['good_credit'] == 0
    high_risk_combo = short_tenure & poor_credit
    refined_prob.loc[high_risk_combo] *= 1.2
    
    # Rule 2: Reduce risk for long-term, high-value customers
    long_tenure = df['days_tenure'] > df['days_tenure'].quantile(0.8)
    high_value = df['curr_ann_amt'] > df['curr_ann_amt'].quantile(0.8)
    loyal_customers = long_tenure & high_value
    refined_prob.loc[loyal_customers] *= 0.85
    
    # Rule 3: Cluster-based adjustments
    if 'Demographics_Cluster' in df.columns:
        # Assuming cluster 6 is high-risk demographic
        high_risk_demo = df['Demographics_Cluster'] == 6
        refined_prob.loc[high_risk_demo] *= 1.1
    
    # Rule 4: Geographic risk adjustments
    if 'Geographic_Cluster' in df.columns:
        # Assuming cluster 3 is high-churn geographic area
        high_risk_geo = df['Geographic_Cluster'] == 3
        refined_prob.loc[high_risk_geo] *= 1.05
    
    # Ensure probabilities stay in [0, 1]
    return np.clip(refined_prob, 0, 1)

def apply_cluster_specific_thresholds(df, optimal_threshold=0.5):
    """Apply different thresholds based on customer segments"""
    refined_pred = np.zeros(len(df))
    
    # Default threshold
    base_pred = (df['predicted_churn_probability'] > optimal_threshold).astype(int)
    
    # More conservative for high-value customers
    if 'curr_ann_amt' in df.columns:
        high_value = df['curr_ann_amt'] > df['curr_ann_amt'].quantile(0.9)
        conservative_threshold = optimal_threshold + 0.1
        conservative_pred = (df['predicted_churn_probability'] > conservative_threshold).astype(int)
        refined_pred = np.where(high_value, conservative_pred, base_pred)
    else:
        refined_pred = base_pred
    
    # More aggressive for budget customers
    if 'Financial_Cluster' in df.columns:
        # Assuming cluster 5 is budget-conscious customers
        budget_customers = df['Financial_Cluster'] == 5
        aggressive_threshold = max(0.3, optimal_threshold - 0.1)
        aggressive_pred = (df['predicted_churn_probability'] > aggressive_threshold).astype(int)
        refined_pred = np.where(budget_customers, aggressive_pred, refined_pred)
    
    return refined_pred

def main():
    parser = argparse.ArgumentParser(description="Refine churn predictions")
    parser.add_argument("--input", required=True, help="Input CSV with predictions")
    parser.add_argument("--output", help="Output CSV (default: <input>_refined.csv)")
    parser.add_argument("--threshold-metric", choices=['f1', 'precision', 'recall'], 
                       default='f1', help="Metric for threshold optimization")
    parser.add_argument("--apply-rules", action='store_true', 
                       help="Apply business rules refinement")
    parser.add_argument("--cluster-thresholds", action='store_true',
                       help="Use cluster-specific thresholds")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    output_path = args.output or args.input.replace('.csv', '_refined.csv')
    
    print(f"Loading {len(df)} predictions from {args.input}")
    
    # Check if we have ground truth for evaluation
    has_ground_truth = 'Churn' in df.columns
    
    if has_ground_truth:
        y_true = df['Churn']
        y_prob = df['predicted_churn_probability']
        
        # Find optimal threshold
        optimal_thresh, best_score = find_optimal_threshold(y_true, y_prob, args.threshold_metric)
        print(f"Optimal threshold for {args.threshold_metric}: {optimal_thresh:.3f} (score: {best_score:.3f})")
        
        # Original performance
        y_pred_orig = df['predicted_churn']
        print("\nOriginal Performance:")
        print(classification_report(y_true, y_pred_orig))
    else:
        optimal_thresh = 0.5
        print("No ground truth available; using default threshold 0.5")
    
    # Apply refinements
    if args.apply_rules:
        print("Applying business rules refinement...")
        df['refined_churn_probability'] = apply_business_rules(df)
    else:
        df['refined_churn_probability'] = df['predicted_churn_probability']
    
    # Apply optimal threshold
    if args.cluster_thresholds:
        print("Applying cluster-specific thresholds...")
        df['refined_churn'] = apply_cluster_specific_thresholds(df, optimal_thresh)
    else:
        df['refined_churn'] = (df['refined_churn_probability'] > optimal_thresh).astype(int)
    
    # Evaluate refined predictions
    if has_ground_truth:
        print("\nRefined Performance:")
        print(classification_report(y_true, df['refined_churn']))
        
        # Confusion matrices
        print("\nConfusion Matrix - Original:")
        print(confusion_matrix(y_true, y_pred_orig))
        print("\nConfusion Matrix - Refined:")
        print(confusion_matrix(y_true, df['refined_churn']))
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nRefined predictions saved to: {output_path}")
    
    # Summary stats
    print(f"\nSummary:")
    print(f"Original churn predictions: {df['predicted_churn'].sum()}")
    print(f"Refined churn predictions: {df['refined_churn'].sum()}")
    print(f"Change: {df['refined_churn'].sum() - df['predicted_churn'].sum():+d}")

if __name__ == "__main__":
    main()