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
import logging
import time


# Local Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import img_grid
import pciam
import translation_refinement
import assemble

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


def mist_function(frames: List[np.ndarray], opts: MistArgs=None, parallel: bool=False) -> None:
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

    mist_start_time = time.time()
    logging.info("Computing all pairwise translations for between images")

    if not parallel or args.disable_mem_cache:
        # if mem cache is off, only use single threaded version to save memory
        translation_computation = pciam.PciamSequential(args)
    else:
        translation_computation = pciam.PciamParallel(args)
    translation_computation.execute(tile_grid)

    # compose the pairwise translations into global positions using MST
    global_positions = translation_refinement.GlobalPositions(tile_grid)
    global_positions.traverse_minimum_spanning_tree()

    output_filename = "{}relative-positions-{}.txt".format(args.output_prefix, args.time_slice)
    tile_grid.write_translations_to_file(os.path.join(args.output_dirpath, output_filename))

    output_filename = "{}global-positions-{}.txt".format(args.output_prefix, args.time_slice)
    global_positions_filepath = os.path.join(args.output_dirpath, output_filename)
    tile_grid.write_global_positions_to_file(global_positions_filepath)

    if args.save_image:
        img_output_filepath = os.path.join(args.output_dirpath, "{}stitched-{}.tif".format(args.output_prefix, args.time_slice))
        assemble.assemble_image_from_grid(global_positions_filepath, tile_grid, img_output_filepath)
        #assemble.assemble_image(global_positions_filepath, args.image_dirpath, img_output_filepath)

    elapsed_time = time.time() - mist_start_time
    logging.info("MIST took {} seconds".format(elapsed_time))
    return
