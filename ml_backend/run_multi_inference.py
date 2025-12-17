#!/usr/bin/env python3
"""
Multi-Model Inference Runner
Runs inference across multiple GBM models and prepares outputs for meta-model training
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDERR: {e.stderr}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run inference across multiple models")
    parser.add_argument("--input", required=True, help="Input CSV for inference")
    parser.add_argument("--output-dir", default="multi_model_outputs", help="Directory for output files")
    parser.add_argument("--python", default="python", help="Python executable to use")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define models and their corresponding files
    models = [
        {
            "name": "37.5k_cluster_v2",
            "model": "churn_model_37.5k_cluster_v2.pkl",
            "features": "model_features_37.5k_cluster_v2.pkl", 
            "metadata": "model_metadata_37.5k_cluster_v2.pkl",
            "output": output_dir / "inference_37.5k_v2.csv"
        },
        {
            "name": "37.5k_cluster_v3", 
            "model": "churn_model_37.5k_cluster_v3.pkl",
            "features": "model_features_37.5k_cluster_v3.pkl",
            "metadata": "model_metadata_37.5k_cluster_v3.pkl", 
            "output": output_dir / "inference_37.5k_v3.csv"
        },
        {
            "name": "37.5k_cluster",
            "model": "churn_model_37.5k_cluster.pkl", 
            "features": "model_features_37.5k_cluster.pkl",
            "metadata": "model_metadata_37.5k_cluster.pkl",
            "output": output_dir / "inference_37.5k.csv"
        },
        {
            "name": "full_cluster",
            "model": "churn_model_full_cluster.pkl",
            "features": "model_features_full_cluster.pkl", 
            "metadata": "model_metadata_full_cluster.pkl",
            "output": output_dir / "inference_full.csv"
        }
    ]
    
    print(f"Input file: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Models to run: {len(models)}")
    
    successful_runs = []
    failed_runs = []
    
    # Run inference for each model
    for model_config in models:
        print(f"\n{'*'*80}")
        print(f"Processing model: {model_config['name']}")
        print(f"{'*'*80}")
        
        # Check if model files exist
        missing_files = []
        for file_type in ["model", "features", "metadata"]:
            if not os.path.exists(model_config[file_type]):
                missing_files.append(model_config[file_type])
        
        if missing_files:
            print(f"SKIPPING {model_config['name']}: Missing files: {missing_files}")
            failed_runs.append({
                "model": model_config['name'],
                "reason": f"Missing files: {missing_files}"
            })
            continue
        
        # Build command
        cmd = [
            args.python, "x.py",
            "--input", args.input,
            "--output", str(model_config["output"]),
            "--model", model_config["model"],
            "--features", model_config["features"], 
            "--metadata", model_config["metadata"]
        ]
        
        # Run inference
        success = run_command(cmd, f"Inference for {model_config['name']}")
        
        if success:
            # Verify output file was created
            if model_config["output"].exists():
                print(f"✓ Output saved: {model_config['output']}")
                successful_runs.append(model_config)
            else:
                print(f"✗ Output file not found: {model_config['output']}")
                failed_runs.append({
                    "model": model_config['name'],
                    "reason": "Output file not created"
                })
        else:
            failed_runs.append({
                "model": model_config['name'], 
                "reason": "Command execution failed"
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Successful runs: {len(successful_runs)}")
    for run in successful_runs:
        print(f"  ✓ {run['name']} → {run['output']}")
    
    print(f"\nFailed runs: {len(failed_runs)}")
    for run in failed_runs:
        print(f"  ✗ {run['model']}: {run['reason']}")
    
    if successful_runs:
        print(f"\nNext step: Use prepare_metamodel_dataset.py to combine these outputs")
        print(f"Output directory: {output_dir.absolute()}")
    
    return len(successful_runs) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)