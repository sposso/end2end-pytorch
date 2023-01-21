'''This script contains the helper functions to use 
the patch classifier in a sliding window fashion and 
get the heatmaps (saliency maps)'''

import numpy as np 


def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

def sweep_img_patches(img, patch_size, stride):
    '''Generate the patches through whole image'''

    nb_row = round(float(img.shape[0] - patch_size)/stride + .49)
    nb_col = round(float(img.shape[1] - patch_size)/stride + .49)
    nb_row = int(nb_row)
    nb_col = int(nb_col)
    sweep_hei = patch_size + (nb_row - 1)*stride
    sweep_wid = patch_size + (nb_col - 1)*stride
    y_gap = int((img.shape[0] - sweep_hei)/2)
    x_gap = int((img.shape[1] - sweep_wid)/2)
    patch_list = []
    for y in range(y_gap, y_gap + nb_row*stride, stride):
        for x in range(x_gap, x_gap + nb_col*stride, stride):
            patch = img[y:y+patch_size, x:x+patch_size].copy()
            patch_list.append(patch.astype('float32'))
    return np.stack(patch_list), nb_row, nb_col
