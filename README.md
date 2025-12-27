
# Deep RL for Tractography

This project investigates the performance of Deep Reinforcement Learning (Soft Actor-Critic) compared to traditional Deterministic algorithms for Diffusive MRI Tractography, specifically focusing on the challenge of "crossing fibers".

## Overview
The codebase facilitates:
1.  **Deterministic Tracking**: A baseline using standard Local Tracking with a maximum angle constraint.
2.  **RL Agent Tracking**: Using a pre-trained SAC (Soft Actor-Critic) agent to navigate fiber bundles.
3.  **Visualization & Analysis**: Scripts to render tractograms and analyze agent decision-making.

## Prerequisites
- Python 3.8+
- [TrackToLearn](https://github.com/GuillaumeTh/TrackToLearn) (included as submodule/folder)
- DIPY
- Nibabel
- Fury (for rendering)
- Matplotlib

### Installation
```bash
pip install -e TrackToLearn
pip install dipy nibabel fury matplotlib
```

## Usage
**Note:** Always run scripts from the **root** directory of the repository.

### 1. Run Baseline (Deterministic)
Generates a deterministic tractogram in the ROI.
```bash
python3 scripts/run_deterministic_roi.py
```
*Output: `deterministic_roi.trk` (and `roi_mask.nii.gz`)*

### 2. Run RL Agent (SAC)
Runs the pre-trained SAC agent in the same ROI.
```bash
python3 scripts/run_sac_roi.py
```
*Output: `sac_roi.trk`*

### 3. Visualize Results
Render the generated tractograms to PNG images.
```bash
python3 scripts/render_tractogram.py deterministic_roi.trk figures/deterministic_roi.png
python3 scripts/render_tractogram.py sac_roi.trk figures/sac_roi.png
```

### 4. Analyze Policy
Generate a histogram of the agent's action directions.
```bash
python3 scripts/plot_action_histogram.py sac_roi.trk figures/action_histogram.png
```

## Project Structure
- `scripts/`: Python scripts for tracking, visualization, and analysis.
- `figures/`: Generated images and plots.
- `notebooks/`: Jupyter notebooks (e.g., `connect.ipynb`).
- `docs/`: PDF documentation and articles.
- `TrackToLearn/`: The core RL tracking library.
- `experiments/`: Training logs and checkpoints.

## Key Files
- `scripts/run_deterministic_roi.py`: Baseline tracking script.
- `scripts/run_sac_roi.py`: RL inference script using `TrackToLearn`.
- `scripts/render_tractogram.py`: Visualization utility using FURY/VTK.

## Dataset
This project uses the ISMRM 2015 Tractography Challenge dataset. Ensure `ISMRM_2015_Tracto_challenge_data/` is present in the root directory.

## References 
References
Christiaens, D., Tournier, J.-D. and Seiberlich, N. (2020) 'Modeling fiber orientations using diffusion MRI', in Quantitative Magnetic Resonance Imaging (Advances in Magnetic Resonance Technology and Applications, Volume 1). Elsevier, pp. 509–534. doi:10.1016/B978–0–12–817057–1.00022–6.​
Du, H., Li, Z., Niyato, D., Kang, J., Xiong, Z., Xu, X., Shen, X. and Kim, D.I. (2023) 'Enabling AI-generated content (AIGC) services in wireless edge networks', arXiv preprint arXiv:2301.03220. doi:10.48550/arXiv.2301.03220.​
Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018) 'Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor', in Proceedings of the 35th International Conference on Machine Learning (ICML 2018), pp. 1861–1870.​
Henderson, F., Abdullah, K.G. and Tsering, D. (2020) 'Tractography and the connectome in neurosurgical applications', Neurosurgical Focus, 48(2), E6.​
Rheault, F., Poulin, P., Valcourt Caron, A., St-Onge, E. and Descoteaux, M. (2020) 'Common misconceptions, hidden biases and modern challenges of dMRI tractography', Journal of Neural Engineering, 17(1), 011001. doi:10.1088/1741–2552/ab6aad.​
Schilling, K.G., Nath, V., Hansen, C., Parvathaneni, P., Blaber, J., Gao, Y., Neher, P., Aydogan, D.B., Shi, Y. and O'Donnell, L.J. (2020) 'Brain connections derived from diffusion MRI tractography can be highly anatomically inaccurate', NeuroImage, 215, 116767.​
Théberge, A., Ferland, G., Soucy, J.-P., Descoteaux, M. and Girard, G. (2021) 'Track-to-Learn: A general framework for tractography with deep reinforcement learning', NeuroImage, 239, 118316.​
