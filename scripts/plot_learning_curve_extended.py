
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_curve():
    reward_path = 'experiments/plots/train_reward.npy'
    entropy_path = 'experiments/plots/entropy.npy'
    
    if not os.path.exists(reward_path):
        print(f"Error: {reward_path} not found.")
        return

    # Load Reward
    r_data = np.load(reward_path)
    r_ep = r_data[:, 0]
    r_val = r_data[:, 1]
    
    # Load Entropy
    if os.path.exists(entropy_path):
        e_data = np.load(entropy_path)
        # Handle pickle if needed (it shouldn't be needed now)
        e_ep = e_data[:, 0]
        e_val = e_data[:, 1]
    else:
        print("Entropy logs not found. Plotting only return.")
        e_ep, e_val = None, None

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Return
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Return', color=color)
    ax1.plot(r_ep, r_val, color=color, label='Average Return')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Entropy
    if e_ep is not None:
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Entropy (Alpha)', color=color)
        ax2.plot(e_ep, e_val, color=color, linestyle='--', label='Entropy')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Learning Curve: Return and Entropy')
    fig.tight_layout()
    
    output_path = 'learning_curve_extended.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_curve()
