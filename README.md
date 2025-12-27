
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
