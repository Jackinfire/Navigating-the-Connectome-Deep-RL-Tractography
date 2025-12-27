
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import nibabel as nib
import torch
import json

from TrackToLearn.experiment.experiment import Experiment
from TrackToLearn.utils.torch_utils import get_device
from TrackToLearn.algorithms.sac_auto import SACAuto
# We can use TrackToLearnTrack logic

# We basically need to instantiate TrackToLearnTrack with correct DTO
# But TrackToLearnTrack parses arguments extensively.
# It might be easier to just construct the DTO and instantiate the class.

from TrackToLearn.runners.ttl_track import TrackToLearnTrack

def run_sac_roi():
    print("Preparing SAC ROI Tracking...")
    
    # Paths
    data_dir = 'ISMRM_2015_Tracto_challenge_data'
    fodf_path = os.path.join(data_dir, 'fodf.nii.gz') # "in_odf"
    roi_mask_path = 'roi_mask.nii.gz' # "in_seed" and "in_mask"
    output_path = 'sac_roi.trk'
    
    model_dir = 'TrackToLearn/models'
    # We need the checkpoint file, but ttl_track expects a folder?
    # args.agent: "Path to the folder containing .pth files."
    
    # Let's construct the DTO (Data Transfer Object) expected by TrackToLearnTrack
    track_dto = {
        'in_odf': fodf_path,
        'in_seed': roi_mask_path,
        'in_mask': os.path.join(data_dir, 'mask.nii.gz'), # Track only within ROI? Or seed in ROI and track full brain?
        # User said "Take a region of a few voxels only". 
        # Usually this implies tracking LOCALLY in that region.
        # But for comparison, if deterministic tracked in ROI mask, SAC should too?
        # Deterministic script: "stopping_criterion = BinaryStoppingCriterion(mask_data)"
        # where mask_data was the FULL mask or ROI mask?
        # In run_deterministic_roi.py:
        # stopping_criterion = BinaryStoppingCriterion(mask_data) -> FULL MASK
        # seeds = seeds_from_mask(roi_data) -> SEEDS IN ROI
        # So deterministic tracked FROM ROI into the WHOLE BRAIN (bounded by full mask).
        
        # SAC should do the same?
        # "Take a region of a few voxels only".
        # If I restrict tracking mask to ROI, streamlines will be very short.
        # The deterministic script used `mask_data` (full brain) for stopping.
        # So it tracks long fibers starting from the ROI.
        
        # I should match that.
        # 'in_mask' argument in ttl_track is "tracking mask".
        # 'in_seed' is seeding mask.
        
        # So:
        'in_mask': os.path.join(data_dir, 'mask.nii.gz'),
        'in_seed': roi_mask_path,
        
        'out_tractogram': output_path,
        'input_wm': False, # Assuming False, check training args. 
        # In sac_auto_train, we didn't specify --input_wm, default is False.
        
        'noise': 0.0,
        'binary_stopping_threshold': 0.1,
        'n_actor': 1024, # How many streamlines? 
        # Deterministic generated ~2 seeds * 1000 voxels = 2000 streamlines.
        # Let's use comparable number. 2000.
        
        'npv': 2, # Seeds per voxel
        'min_length': 20.0, # Match training
        'max_length': 200.0,
        
        'compress': 0.0,
        'sh_basis': ['descoteaux07'], # Need list?
        'save_seeds': False,
        
        'agent': model_dir,
        'hyperparameters': os.path.join(model_dir, 'hyperparameters.json'),
        
        'rng_seed': 1337
    }
    
    # Check if files exist
    if not os.path.exists(track_dto['in_odf']):
        print(f"Error: {track_dto['in_odf']} not found.")
        return
    if not os.path.exists(track_dto['in_seed']):
        print(f"Error: {track_dto['in_seed']} not found.")
        return

    print("Running Tracker...")
    tracker_experiment = TrackToLearnTrack(track_dto)
    tracker_experiment.run()
    print("Done!")

if __name__ == "__main__":
    run_sac_roi()
