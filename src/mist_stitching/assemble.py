import os
import numpy as np
import skimage.io

def _get_global_xy(fname):
    if not os.path.exists(fname):
        raise RuntimeError('Missing global positions file: {}'.format(fname))

    # load the global positions for each image
    img_names = list()
    pixel_x_position = list()
    pixel_y_position = list()
    with open(fname, 'r') as fh:
        for line in fh:
            line = line.strip()
            toks = line.split(';')

            # handle file name loading
            fn_tok = toks[0]
            fn = fn_tok.split(':')[1].strip()
            img_names.append(fn)

            # handle the position loading
            pos_tok = toks[2]
            pos_pair = pos_tok.split(':')[1].strip()
            pos_pair = pos_pair.replace(')', '')
            pos_pair = pos_pair.replace('(', '')
            pos_pairs = pos_pair.split(',')
            x = int(pos_pairs[0].strip())
            y = int(pos_pairs[1].strip())
            pixel_x_position.append(x)
            pixel_y_position.append(y)
    
    return pixel_x_position, pixel_y_position, img_names

def assemble_image_from_grid(global_positions_filepath, tile_grid, output_filepath):
    """ build the stitched image from the tile grid and global positions """
    
    parent, fn = os.path.split(output_filepath)
    if not os.path.exists(parent):
        os.makedirs(parent)

    # Get the global positions and image names from the file
    pixel_x_position, pixel_y_position, img_names = _get_global_xy(global_positions_filepath)

    # Get size and type of first tile
    first_tile = tile_grid.tiles[0][0].get_image()
    tile_shape = first_tile.shape
    n_channels = 1
    if len(tile_shape) == 3:
        n_channels = tile_shape[2]
    tile_h = tile_shape[0]
    tile_w = tile_shape[1]
    
    stitched_img_h = tile_h + np.max(pixel_y_position)
    stitched_img_w = tile_w + np.max(pixel_x_position)

    # creating blank image
    print('Creating blank stitched image of size: ({}, {}, {})'.format(stitched_img_h, stitched_img_w, n_channels))
    if n_channels == 1:
        stitched_img = np.zeros((stitched_img_h, stitched_img_w), dtype=first_tile.dtype)
    else:
        stitched_img = np.zeros((stitched_img_h, stitched_img_w, n_channels), dtype=first_tile.dtype)


    for r in range(tile_grid.args.grid_height):
        for c in range(tile_grid.args.grid_width):
            img_tile = tile_grid.get_tile(r, c).get_image()
            if img_tile is None:
                continue
            if img_tile.shape != tile_shape:
                raise RuntimeError('All images must be the same shape. Image {} is {}, expected {}'.format(fn, img_tile.shape, tile_shape))
            if img_tile.dtype != first_tile.dtype:
                raise RuntimeError('Img {} has type: {}, expected {}.'.format(fn, img_tile.dtype, first_tile.dtype))
            
            idx = r * tile_grid.args.grid_width + c
            fn = img_names[idx]
            x = pixel_x_position[idx]
            y = pixel_y_position[idx]
            print('Img {}/{}. Placing {} at ({}, {})'.format(idx, len(img_names), fn, x, y))

            if n_channels == 1:
                stitched_img[y:y+tile_h, x:x+tile_w] = img_tile
            else:
                stitched_img[y:y+tile_h, x:x+tile_w, :] = img_tile
    
    print('Saving stitched image to disk')
    skimage.io.imsave(output_filepath, stitched_img, check_contrast=False)


def assemble_image(global_positions_filepath, images_dirpath, output_filepath):

    parent, fn = os.path.split(output_filepath)
    if not os.path.exists(parent):
        os.makedirs(parent)

    # Get the global positions and image names from the file
    pixel_x_position, pixel_y_position, img_names = _get_global_xy(global_positions_filepath)

    # verify that all images exist
    if not os.path.exists(images_dirpath):
        raise RuntimeError('Images directory does not exist: {}'.format(images_dirpath))

    for fn in img_names:
        if not os.path.exists(os.path.join(images_dirpath, fn)):
            raise RuntimeError('Image {} expected based on global positions file, but its missing from the image directory.'.format(fn))

    # compute how large of an output image will be required.
    first_tile = skimage.io.imread(os.path.join(images_dirpath, img_names[0]))
    tile_shape = first_tile.shape
    n_channels = 1
    if len(tile_shape) == 3:
        n_channels = tile_shape[2]
    tile_h = tile_shape[0]
    tile_w = tile_shape[1]

    stitched_img_h = tile_h + np.max(pixel_y_position)
    stitched_img_w = tile_w + np.max(pixel_x_position)

    # creating blank image
    print('Creating blank stitched image of size: ({}, {}, {})'.format(stitched_img_h, stitched_img_w, n_channels))
    if n_channels == 1:
        stitched_img = np.zeros((stitched_img_h, stitched_img_w), dtype=first_tile.dtype)
    else:
        stitched_img = np.zeros((stitched_img_h, stitched_img_w, n_channels), dtype=first_tile.dtype)

    for i in range(0, len(img_names)):
        fn = img_names[i]
        x = pixel_x_position[i]
        y = pixel_y_position[i]
        print('Img {}/{}. Placing {} at ({}, {})'.format(i, len(img_names), fn, x, y))
        tile = skimage.io.imread(os.path.join(images_dirpath, fn))
        if tile.shape != tile_shape:
            raise RuntimeError('All images must be the same shape. Image {} is {}, expected {}'.format(fn, tile.shape, tile_shape))
        if tile.dtype != first_tile.dtype:
            raise RuntimeError('Img {} has type: {}, expected {}.'.format(fn, tile.dtype, first_tile.dtype))

        if n_channels == 1:
            stitched_img[y:y+tile_h, x:x+tile_w] = tile
        else:
            stitched_img[y:y+tile_h, x:x+tile_w, :] = tile

    print('Saving stitched image to disk')
    skimage.io.imsave(output_filepath, stitched_img, plugin=None, tile=(1024, 1024), check_contrast=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to assemble MIST stitched image')
    parser.add_argument('--global-positions-filepath', type=str, required=True, help='Filepath to the global positions file generated by MIST.')
    parser.add_argument('--images-dirpath', type=str, required=True, help='Dirpath (directory) where the source images exists.')
    parser.add_argument('--output-filepath', type=str, required=True, help='Filepath where to save the resulting stitched image.')

    args = parser.parse_args()
    global_positions_filepath = args.global_positions_filepath
    images_dirpath = args.images_dirpath
    output_filepath = args.output_filepath

    assemble_image(global_positions_filepath, images_dirpath, output_filepath)