
import numpy as np
import os

files = [
    'train_reward.npy', 
    'actor_loss.npy', 
    'critic_loss.npy',
    'length_reward.npy',
    'entropy.npy'
]

print("Checking npy files in experiments/plots/")
for f in files:
    path = os.path.join('experiments/plots', f)
    if os.path.exists(path):
        try:
            data = np.load(path)
            print(f"{f}: Shape {data.shape}")
            if len(data) > 0:
                print(f"  Last 3 points: {data[-3:]}")
            else:
                print("  Empty")
        except Exception as e:
            print(f"{f}: Error {e}")
    else:
        print(f"{f}: Not found")
