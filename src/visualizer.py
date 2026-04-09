import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_matrix(matrix, title="Matrix Heatmap", cmap="viridis", save_path=None):
    """
    Plots a heatmap of a given matrix.
    Args:
        matrix: numpy array or torch tensor.
        title: Title of the plot.
        cmap: Colormap to use.
        save_path: If provided, saves the figure to this path.
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_plop_steps(input_z, norm_z, norm_w, projection, nfn_score, output_dir="viz_output"):
    """
    Plots the 5 steps of the PLoP algorithm.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    steps = [
        (input_z, "Step 1: Raw Input (z)", "magma"),
        (norm_z, "Step 2: Normalized Input (z_tilde)", "viridis"),
        (norm_w, "Step 3: Normalized Weight (W_tilde)", "plasma"),
        (projection, "Step 4: Projection Result (Wz^T)", "inferno")
    ]
    
    for i, (mat, title, cmap) in enumerate(steps):
        path = os.path.join(output_dir, f"step_{i+1}_matrix.png")
        plot_matrix(mat, title=title, cmap=cmap, save_path=path)
    
    # Final score as a simple text/bar plot or just a console log for now
    print(f"\nFinal Calculated NFN Score for this sequence: {nfn_score:.4f}")
