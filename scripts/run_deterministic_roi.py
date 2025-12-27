
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nibabel as nib
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion

def run_deterministic_roi():
    print("Loading data...")
    data_dir = 'ISMRM_2015_Tracto_challenge_data'
    fodf_path = os.path.join(data_dir, 'fodf.nii.gz')
    mask_path = os.path.join(data_dir, 'mask.nii.gz')
    roi_mask_output_path = 'roi_mask.nii.gz'
    output_path = 'deterministic_roi.trk'
    
    fodf_img = nib.load(fodf_path)
    fodf_data = fodf_img.get_fdata()
    affine = fodf_img.affine
    
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)
    
    # Define ROI: Center of the brain (center of mask)
    print("Computing ROI...")
    coords = np.argwhere(mask_data)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    center = (min_c + max_c) // 2
    
    roi_size = 10 # 10 voxels radius? Or 10x10x10 total? 
    # "region of a few voxels only". 10x10x10 is 1000 voxels, manageable.
    # Let's do +/- 5 voxels from center
    
    roi_data = np.zeros_like(mask_data)
    
    # Ensure bounds
    s_x = slice(max(0, center[0]-5), min(mask_data.shape[0], center[0]+5))
    s_y = slice(max(0, center[1]-5), min(mask_data.shape[1], center[1]+5))
    s_z = slice(max(0, center[2]-5), min(mask_data.shape[2], center[2]+5))
    
    roi_data[s_x, s_y, s_z] = mask_data[s_x, s_y, s_z]
    
    # Save ROI mask
    print(f"Saving ROI mask to {roi_mask_output_path}...")
    roi_img = nib.Nifti1Image(roi_data.astype(np.uint8), affine)
    nib.save(roi_img, roi_mask_output_path)
    
    print("Computing ODF from SH...")
    sphere = get_sphere('repulsion724')
    # Use SH order 8 and descoteaux07 basis as per README/generate_peaks
    odf = sh_to_sf(fodf_data, sphere, sh_order=8, basis_type='descoteaux07')
    
    print("Initializing DirectionGetter...")
    # DeterministicMaximumDirectionGetter takes PMF (ODF values)
    dg = DeterministicMaximumDirectionGetter.from_pmf(odf, max_angle=15., sphere=sphere)
    
    print("Initializing Stopping Criterion...")
    # Stop if out of mask (use full mask for stopping, but seed in ROI)
    stopping_criterion = BinaryStoppingCriterion(mask_data)

    print("Generating Seeds in ROI...")
    seeds = utils.seeds_from_mask(roi_data, affine, density=2)
    
    print(f"Tracking with {len(seeds)} seeds...")
    tracker = LocalTracking(dg, stopping_criterion, seeds, affine, step_size=0.75)
    
    print("Tracking...")
    streamlines = Streamlines()
    for i, sl in enumerate(tracker):
        streamlines.append(sl)
    print(f"Generated {len(streamlines)} streamlines.")
    
    print(f"Saving to {output_path}...")
    sft = StatefulTractogram(streamlines, roi_img, Space.RASMM)
    save_tractogram(sft, output_path, bbox_valid_check=False)
    print("Done!")

if __name__ == "__main__":
    run_deterministic_roi()
