import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # pip install mpl_toolkits if needed

# Load tractogram
sft = nib.streamlines.load("/Users/ommahajan/Desktop/Year_4/Reinforcement Learning for Bioengineers/Coursework/TrackToLearn/output_ismrm.trk")
streamlines_all = list(sft.streamlines)

# --- Subsample 20% of streamlines ---
fraction = 0.5
n_total = len(streamlines_all)
n_keep = max(1, int(fraction * n_total))
idx_keep = random.sample(range(n_total), n_keep)
streamlines = [streamlines_all[i] for i in idx_keep]  # 20% subset [web:196]

# Slice parameters
z0 = 85.15529        # mm
thickness = 2.0 # mm

xs, ys = [], []
cols = []

def orient2rgb(v):
    """
    v: (N, 3) direction vectors.
    Standard direction-encoded color: R=|x|, G=|y|, B=|z|.
    """
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    v_norm = v / n
    v_abs = np.abs(v_norm)
    r = v_abs[:, 0]
    g = v_abs[:, 1]
    b = v_abs[:, 2]
    rgb = np.stack([r, g, b], axis=1)
    return np.clip(rgb, 0, 1)

# Collect slice points and colors from subsampled streamlines
for sl in streamlines:
    if sl.shape[0] < 2:
        continue

    dirs = np.diff(sl, axis=0)
    pts  = sl[:-1]

    z = pts[:, 2]
    mask = np.abs(z - z0) < thickness / 2.0
    pts_slice = pts[mask]
    dirs_slice = dirs[mask]

    if pts_slice.shape[0] == 0:
        continue

    xs.extend(pts_slice[:, 0])
    ys.extend(pts_slice[:, 1])
    cols.append(orient2rgb(dirs_slice))

if len(xs) == 0:
    raise RuntimeError("No points in this slice; adjust z0/thickness.")

xs = np.array(xs)
ys = np.array(ys)
cols = np.vstack(cols)

fig, ax = plt.subplots(figsize=(5, 5))

# Main scatter (no title)
ax.scatter(xs, ys, s=2, c=cols, alpha=0.8)
ax.invert_yaxis()
ax.axis("equal")
ax.axis("off")

# --- Direction colourwheel inset ---
res = 101
theta = np.linspace(0, 2 * np.pi, res)
r = np.linspace(0, 1, res)
R, T = np.meshgrid(r, theta)
X = R * np.cos(T)
Y = R * np.sin(T)
Z = np.zeros_like(X)

vecs = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
wheel_rgb = orient2rgb(vecs).reshape(res, res, 3)
ax.set_title("Human Brain Connectome") 

# ax_inset = inset_axes(ax, width="25%", height="25%", loc="lower left",
#                       borderpad=0.5)
# ax_inset.imshow(wheel_rgb, origin="lower", extent=[-1, 1, -1, 1])
# ax_inset.set_xticks([])
# ax_inset.set_yticks([])
# ax_inset.set_xlabel("L–R", fontsize=6)
# ax_inset.set_ylabel("A–P", fontsize=6)

# # after creating ax_inset and showing wheel_rgb
# ax_inset.arrow(0, 0, 0.8, 0, head_width=0.1, head_length=0.1,
#                fc="red", ec="red")    # L–R
# ax_inset.arrow(0, 0, 0, 0.8, head_width=0.1, head_length=0.1,
#                fc="green", ec="green")  # A–P

# ax_inset.text(0.9, 0.0, "L/R", color="red", fontsize=6, ha="left", va="center")
# ax_inset.text(0.0, 0.9, "A/P", color="green", fontsize=6, ha="center", va="bottom")

plt.tight_layout()
plt.show()
