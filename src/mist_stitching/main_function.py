"""
Main function for mist stitching.

This fie allows calling the MIST stitching process from another Python script.

This is meant for speed, so the entire image grid is passed in as an argument, rather than
reloading the images from disk.

"""
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import argparse


# Local Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import img_grid

@dataclass
class MistArgs:
    image_dirpath: str
    output_dirpath: str
    output_prefix: str = 'img-'
    save_image: bool = True
    disable_mem_cache: bool = False

    # stage model parameters
    stage_repeatability: Optional[float] = None
    horizontal_overlap: Optional[float] = None
    vertical_overlap: Optional[float] = None
    overlap_uncertainty: float = 3.0
    valid_correlation_threshold: float = 0.5
    time_slice: int = 0

    # advanced parameters
    translation_refinement_method: str = 'SINGLEHILLCLIMB'
    num_hill_climbs: int = 16
    num_fft_peaks: int = 2


def mist_function(frames: List[np.ndarray], opts: MistArgs=None):
    """
    Function to be called from another Python script to run MIST stitching.
    The image grid is passed in as a list of numpy arrays.
    The opts argument is a MistArgs dataclass instance holding the parameters.

    Need to know how the grid is structured. Ideally this would be a 2D list of the images

    """
    if opts is None:
        opts = MistArgs(image_dirpath="", output_dirpath="")

    args = argparse.Namespace()
    for field in opts.__dataclass_fields__:
        setattr(args, field, getattr(opts, field))

    # Determine the number of rows and columns in frames
    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
        args.grid_height = len(frames)
        args.grid_width = len(frames[0])
    else:
        raise ValueError("frames should be a 2D list (list of lists) of numpy arrays.")

    # build the grid representation
    tile_grid = img_grid.TileGirdFrames(frames, args)
    tile_grid.print_names()
    
    return
