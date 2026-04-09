import torch
import torch.nn as nn
import numpy as np
from src.visualizer import plot_plop_steps

def run_plop_trace_demo():
    print("Initializing PLoP Algorithm Simulation Trace...")
    
    # 1. Setup Dummy Dimensions
    seq_len = 8
    hidden_dim = 16
    out_dim = 16
    
    # 2. Generate Synthetic Data
    # Raw Input z [seq_len, hidden_dim]
    z = torch.randn(seq_len, hidden_dim) * 2.0
    
    # Weight Matrix W [out_dim, hidden_dim]
    W = torch.randn(out_dim, hidden_dim) * 0.5
    
    # 3. Step-by-Step PLoP Logic
    
    # Step 1: Raw Input (z already exists)
    
    # Step 2: Normalize Input (z_tilde)
    z_norm = torch.norm(z, dim=1, keepdim=True)
    z_tilde = z / (z_norm + 1e-8)
    
    # Step 3: Normalize Weight (W_tilde)
    # Using Frobenius norm average normalization as in metrics.py
    W_f_norm = (W**2).mean().sqrt()
    W_tilde = W / (W_f_norm + 1e-8)
    
    # Step 4: Projection Result (Wz^T)
    # MM: [seq_len, hidden_dim] x [hidden_dim, out_dim] -> [seq_len, out_dim]
    Wz = torch.mm(z_tilde, W_tilde.t())
    
    # Step 5: Calculate Actual Score
    actual_score = torch.norm(Wz, dim=1).mean().item() / np.sqrt(hidden_dim)
    
    # Random Baseline for NFN
    z_random = torch.randn_like(z_tilde)
    z_random_norm = torch.norm(z_random, dim=1, keepdim=True)
    z_random_tilde = z_random / (z_random_norm + 1e-8)
    Wz_random = torch.mm(z_random_tilde, W_tilde.t())
    random_score = torch.norm(Wz_random, dim=1).mean().item() / np.sqrt(hidden_dim)
    
    nfn_score = actual_score / random_score
    
    # 4. Generate Visualizations
    print("Generating step-by-step visualizations...")
    plot_plop_steps(
        input_z=z, 
        norm_z=z_tilde, 
        norm_w=W_tilde, 
        projection=Wz, 
        nfn_score=nfn_score,
        output_dir="plop_viz_demo"
    )
    
    print("\nSimulation Complete. Check the 'plop_viz_demo' folder for matrix heatmaps.")

if __name__ == "__main__":
    run_plop_trace_demo()
