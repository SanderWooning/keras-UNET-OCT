import math
import sys
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

# Required if you want to save a cumulative cost image in
# e.g. pixels with floats
# import SimpleITK as sitk


# only does left-to-right minimumcostpath
# assumes square pixels...
def min_cost_path_octa(costimage, step=2):

    # fill lengths array
    lengths = np.ndarray((2*step+1))
    for i in range(-step, step+1):
        lengths[step+i] = math.sqrt(i*i+1)

    # create cumulative cost image and
    # initialize the first column:
    cumcostimage = np.zeros(costimage.shape)
    cumcostimage[:, 0] = costimage[:, 0]

    # loop over x (from left to right)
    for i in range(1, costimage.shape[1]):

        # loop over j
        for j in range(costimage.shape[0]):

            # compute valid range for s
            min_s = max(0, j-step)
            max_s = min(cumcostimage.shape[0], j+step+1)

            # current costs
            cur_cost = costimage[j,i]

            # compute all new costs in one loop
            new_costs = (cumcostimage[min_s:max_s, i-1] +
                         np.multiply(lengths[min_s-j+step:max_s-j+step],
                                     costimage[min_s:max_s, i-1]+cur_cost))

            # update cumulative cost image
            cumcostimage[j, i] = np.amin(new_costs)

    return cumcostimage


# only does left-to-right minimumcostpath
# assumes square pixels
def min_cost_2_path_octa(costimage_0, costimage_1,
                         step_0=3, step_1=3,
                         mindist=6, maxdist=60,
                         large=1e5):

    assert costimage_0.shape == costimage_1.shape
    assert len(costimage_0.shape) == 2

    # get vars for image dimension
    xdim = costimage_0.shape[1]
    ydim = costimage_0.shape[0]

    # lengths for stepping in j (i.e. distance
    # between pixel centers)
    # lengths_r[0] is for -step, lengths_r[step] is for 0,
    # lengths_r[2*step] is for +step
    lengths_r = np.ndarray((2*step_0+1))
    for i in range(-step_0, step_0+1):
        lengths_r[step_0+i] = math.sqrt(i*i+1)

    # lengths for stepping in k
    lengths_s = np.ndarray((2*step_1+1))
    for i in range(-step_1, step_1+1):
        lengths_s[step_1+i] = math.sqrt(i*i+1)

    # create cumulative cost image, initialize first slice
    cumcostimage = np.zeros((ydim, ydim, xdim))
    cumcostimage[:, :, 0] = (costimage_0[:, 0].reshape((1, ydim)) +
                             costimage_1[:, 0].reshape((ydim, 1)))

    # exclude invalid ranges for j and k: k must be larger than j,
    # difference between k and j must at least be mindist, and at
    # most be maxdist
    for j in range(ydim):
        k_low = min(j+mindist, ydim)
        k_high = min(j+maxdist, ydim)
        # exclude range of positions for k that are not valid
        cumcostimage[0:k_low, j, 0] = large
        cumcostimage[k_high:ydim, j, 0] = large

    # loop over x (from left to right)
    for i in range(1, xdim):

        # loop over j (position of top contour)
        for j in range(ydim):

            # k runs from j+mindist .. j+maxdist
            k_low = min(j+mindist, ydim)
            k_high = min(j+maxdist, ydim)
            # exclude range of 2nd contour positions
            # that are not valid for this value of j
            cumcostimage[0:k_low, j, i] = large
            cumcostimage[k_high:ydim, j, i] = large

            # compute r, which indicates valid range for j
            # (j is index of top contour)
            min_r = max(0, j-step_0)
            max_r = min(ydim, j+step_0+1)
            cur_cost_r = costimage_0[j, i]

            # get costs for this value of j
            cost_img_0_slice = costimage_0[min_r:max_r, i-1]
            add_costs_j = np.multiply(lengths_r[min_r-j+step_0:max_r-j+step_0],
                                      cost_img_0_slice+cur_cost_r)

            # loop over k (position of bottom contour)
            for k in range(k_low, k_high):

                # compute s, which indicates valid range for k
                min_s = max(0, k-step_1)
                max_s = min(ydim, k+step_1+1)
                cur_cost_s = costimage_1[k, i]

                # get path costs for this value of k
                cost_img_1_slice = costimage_1[min_s:max_s, i-1]
                add_costs_k = np.multiply(lengths_s[min_s-k+step_1:max_s-k+step_1],
                                          cost_img_1_slice+cur_cost_s)

                # combine costs of j and k, and add the cumulative costs from
                # previous slice
                cum_cost_slice = cumcostimage[min_s:max_s, min_r:max_r, i-1]
                total_costs = cum_cost_slice + add_costs_j + add_costs_k.reshape((max_s-min_s, 1))

                # update costs based on minimum of previous determind costs
                min_cost = np.amin(total_costs)
                cumcostimage[k, j, i] = min_cost

    return cumcostimage


# Traces back a path, given a costimage
# Runs from right to left (i.e. from high to
# low x-index), and returns an array of
# tuples with the positions
def trace_back_2D(cumcostimage, step=2):

    # initialize array
    path = []

    # get first minimum (in last slice/column)
    max_i = cumcostimage.shape[1]-1
    minpos = int(np.argmin(cumcostimage[:, max_i]))
    path.append((max_i, minpos))

    # for all previous slices/columns
    for i in range(max_i-1, -1, -1):
        # determine search range, based on current
        # position and the step size
        min_j = max(0, minpos-step)
        max_j = min(cumcostimage.shape[0], minpos+step+1)
        # get the candidates from the column
        candidates = cumcostimage[min_j:max_j, i]
        # get the minimum candidate
        min_cand = int(np.argmin(candidates))
        # convert back to the absolute position
        # in the image
        minpos = min_j+min_cand
        # add point to list
        path.append((i,minpos))

    return path


# Traces back two paths, given a costimage
# Runs from right to left (i.e. from high to
# low x-index), and returns two arrays of
# tuples with the positions
def trace_back_3D(cumcostimage, step=2):

    # initialize arrays
    path_0 = []
    path_1 = []

    # get first minimum (in last slice)
    max_i = cumcostimage.shape[2]-1
    last_slice = cumcostimage[:, :, max_i]
    index = np.argmin(last_slice)
    (minpos) = np.unravel_index(index, cumcostimage[:, :, max_i].shape)

    # add to list
    path_0.append((max_i, minpos[1]))
    path_1.append((max_i, minpos[0]))

    # for all previous slices/columns
    for i in range(max_i-1, -1, -1):
        # determine search range, based on current
        # position and the step size
        min_j = max(0, minpos[1]-step)
        max_j = min(cumcostimage.shape[1], minpos[1]+step+1)
        min_k = max(0, minpos[0]-step)
        max_k = min(cumcostimage.shape[0], minpos[0]+step+1)
        # get candidates
        candidates = cumcostimage[min_k:max_k, min_j:max_j, i]
        # get position of minimum in candidates
        index = np.argmin(candidates)
        (minpos) = np.unravel_index(index, candidates.shape)
        # convert to absolute positions in the image
        minpos = (minpos[0]+min_k, minpos[1]+min_j)
        # add points to lists
        path_0.append((i, minpos[1]))
        path_1.append((i, minpos[0]))

    # return results
    return path_0, path_1

# Normalizes an image (i.e. give it a range from 0 to 1)
def normalize_image(image):
    amin = np.amin(image)
    amax = np.amax(image)
    return (image - amin) / (amax-amin)
