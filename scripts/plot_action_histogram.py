
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
import matplotlib.pyplot as plt

def plot_action_histogram(trk_path, output_path):
    print(f"Loading {trk_path}...")
    try:
        # Load tractogram (without reference to avoid affine issues if not needed for simple diffs)
        # Using 'same' to keep it simple, or we can just read streamlines
        sft = load_tractogram(trk_path, 'same')
        streamlines = sft.streamlines
    except Exception as e:
        print(f"Error loading tractogram: {e}")
        return

    print(f"Computing directions for {len(streamlines)} streamlines...")
    all_directions = []

    for sl in streamlines:
        if len(sl) < 2:
            continue
        # Compute vectors: p[i+1] - p[i]
        diffs = sl[1:] - sl[:-1]
        # Normalize to get unit vectors (directions)
        norms = np.linalg.norm(diffs, axis=1)[:, None]
        # Avoid division by zero
        norms[norms == 0] = 1e-9
        dirs = diffs / norms
        all_directions.append(dirs)

    if not all_directions:
        print("No valid streamlines found.")
        return

    # Concatenate all directions
    all_directions = np.concatenate(all_directions, axis=0)
    print(f"Total steps analyzed: {len(all_directions)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, ax in enumerate(axes):
        ax.hist(all_directions[:, i], bins=50, color=colors[i], alpha=0.7, density=True)
        ax.set_title(f'{components[i]} Component Distribution')
        ax.set_xlabel('Value (Cosine)')
        ax.set_ylabel('Density')
        ax.set_xlim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Action (Direction) Histogram - {len(streamlines)} Streamlines", fontsize=16)
    plt.tight_layout()
    
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path)
    print("Done.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 plot_action_histogram.py <input.trk> <output.png>")
    else:
        plot_action_histogram(sys.argv[1], sys.argv[2])
