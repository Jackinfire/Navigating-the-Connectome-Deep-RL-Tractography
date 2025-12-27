import numpy as np
import matplotlib.pyplot as plt

# Grid size
height, width = 5, 5

# Create empty grid
# 0 = empty, 1 = start, 2 = goal, 3 = obstacle
grid = np.zeros((height, width))

start = (0, 0)
goal = (height - 1, width - 1)

# Middle cell as obstacle
obstacle = (height // 2, width // 2)

grid[start] = 1
grid[goal] = 2
grid[obstacle] = 3

fig, ax = plt.subplots(figsize=(4, 4))

# Use a discrete colormap: empty (light), start (blue), goal (red), obstacle (black)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["lightgrey", "lightblue", "lightcoral", "black"])

ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=3)

# Draw grid lines
ax.set_xticks(np.arange(-0.5, width, 1))
ax.set_yticks(np.arange(-0.5, height, 1))
ax.grid(color="black", linewidth=0.8)

# Remove tick labels for a clean schematic
ax.set_xticklabels([])
ax.set_yticklabels([])

# Labels
ax.text(start[1], start[0], "S", ha="center", va="center",
        color="navy", fontsize=14, fontweight="bold")
ax.text(goal[1], goal[0], "G", ha="center", va="center",
        color="darkred", fontsize=14, fontweight="bold")

ax.set_title("Simplified Grid World with Obstacle")

plt.tight_layout()
plt.show()
