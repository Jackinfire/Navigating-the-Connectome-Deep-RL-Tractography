
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

def run_deterministic():
    print("Loading data...")
    data_dir = 'ISMRM_2015_Tracto_challenge_data'
    fodf_path = os.path.join(data_dir, 'fodf.nii.gz')
    mask_path = os.path.join(data_dir, 'mask.nii.gz')
    output_path = 'deterministic.trk'
    
    fodf_img = nib.load(fodf_path)
    fodf_data = fodf_img.get_fdata()
    affine = fodf_img.affine
    
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)
    
    print("Computing ODF from SH...")
    sphere = get_sphere('repulsion724')
    # Use SH order 8 and descoteaux07 basis as per README/generate_peaks
    odf = sh_to_sf(fodf_data, sphere, sh_order=8, basis_type='descoteaux07')
    
    print("Initializing DirectionGetter...")
    # DeterministicMaximumDirectionGetter takes PMF (ODF values)
    dg = DeterministicMaximumDirectionGetter.from_pmf(odf, max_angle=30., sphere=sphere)
    
    print("Initializing Stopping Criterion...")
    stopping_criterion = BinaryStoppingCriterion(mask_data)

    print("Generating Seeds...")
    # Density 1 or 2. Let's use 2 to match RL.
    seeds = utils.seeds_from_mask(mask_data, affine, density=2)
    
    print(f"Tracking with {len(seeds)} seeds...")
    # step_size=0.75 mm
    tracker = LocalTracking(dg, stopping_criterion, seeds, affine, step_size=0.75)
    
    print("Tracking...")
    streamlines = Streamlines()
    for i, sl in enumerate(tracker):
        streamlines.append(sl)
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} streamlines...", end='\r')
    print(f"\nGenerated {len(streamlines)} streamlines.")
    
    print(f"Saving to {output_path}...")
    sft = StatefulTractogram(streamlines, mask_img, Space.RASMM)
    save_tractogram(sft, output_path, bbox_valid_check=False)
    print("Done!")

if __name__ == "__main__":
    run_deterministic()
