
import argparse
import sys
# Suppress heavy warnings from dipy/vtk
import warnings
warnings.filterwarnings("ignore")

import nibabel as nib
import numpy as np
from dipy.viz import window, actor
from dipy.io.streamline import load_tractogram

def main():
    parser = argparse.ArgumentParser(description="Render tractogram to PNG")
    parser.add_argument("trk_file", help="Path to TRK file")
    parser.add_argument("output_file", help="Path to output PNG file")
    parser.add_argument("--count", type=int, default=10000, help="Number of streamlines to render")
    args = parser.parse_args()

    print(f"Loading {args.trk_file}...")
    try:
        # Load tractogram
        sft = load_tractogram(args.trk_file, 'same')
        streamlines = sft.streamlines
        
        n_streamlines = len(streamlines)
        print(f"Total streamlines: {n_streamlines}")
        
        # Subsample if needed
        if n_streamlines > args.count:
            print(f"Subsampling to {args.count}...")
            # Random choice
            indices = np.random.choice(n_streamlines, args.count, replace=False)
            streamlines = streamlines[indices]
        
        print("Building scene...")
        scene = window.Scene()
        scene.background((1, 1, 1)) # White background
        
        # Color by orientation (standard)
        streamline_actor = actor.line(streamlines)
        scene.add(streamline_actor)
        
        # Camera adjustments
        # We might need to orient the camera. 
        # By default it looks at the center.
        scene.reset_camera()
        scene.zoom(1.2)
        
        print(f"Saving render to {args.output_file}...")
        window.record(scene, out_path=args.output_file, size=(800, 800))
        print("Done.")
        
    except Exception as e:
        print(f"Error rendering: {e}")
        #sys.exit(1)

if __name__ == "__main__":
    main()
