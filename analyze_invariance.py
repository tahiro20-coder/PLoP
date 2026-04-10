import json
import os
import argparse
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_nfn_vector(metrics_dict):
    # Sort keys to ensure consistent vector order
    keys = sorted(metrics_dict.keys())
    return np.array([metrics_dict[k]['nfn'] for k in keys]), keys

def analyze_invariance(results_dir, model_name, datasets):
    vectors = {}
    
    for ds in datasets:
        # Look for aggregated results or raw metrics
        filename = f"{model_name}_{ds}_aggregated_by_type.json"
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            # Fallback to standard metrics
            filename = f"{model_name}_{ds}_metrics.json"
            filepath = os.path.join(results_dir, filename)
            
        if os.path.exists(filepath):
            data = load_metrics(filepath)
            vec, keys = extract_nfn_vector(data)
            vectors[ds] = vec
            print(f"Loaded {ds} vector (dim: {len(vec)})")
        else:
            print(f"Warning: Results not found for {ds} at {filepath}")

    if len(vectors) < 2:
        print("Error: Need at least 2 datasets to compare.")
        return

    ds_list = list(vectors.keys())
    print("\n" + "="*60)
    print(f" Task-Invariance Analysis for {model_name} ".center(60, '='))
    print("="*60)
    
    for i in range(len(ds_list)):
        for j in range(i + 1, len(ds_list)):
            ds1, ds2 = ds_list[i], ds_list[j]
            v1, v2 = vectors[ds1], vectors[ds2]
            
            # Pad vectors if they differ in length (should not happen if same model)
            if len(v1) != len(v2):
                print(f"Warning: Vector length mismatch between {ds1} and {ds2}. Analysis might be skewed.")
                continue
            
            cosine = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
            corr, _ = pearsonr(v1, v2)
            
            print(f"{ds1:>12} vs {ds2:<12} | Cosine Sim: {cosine:.4f} | Pearson Corr: {corr:.4f}")
            
            if cosine > 0.98 or corr > 0.98:
                print(f"  [!] HIGH INVARIANCE DETECTED between {ds1} and {ds2}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze task-invariance of PLoP metrics.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing JSON results')
    parser.add_argument('--model_basename', type=str, required=True, help='Basename of the model (e.g., Phi-3-mini-4k-instruct)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['medic', 'math', 'code', 'history'], help='Datasets to compare')
    args = parser.parse_args()
    
    analyze_invariance(args.results_dir, args.model_basename, args.datasets)
