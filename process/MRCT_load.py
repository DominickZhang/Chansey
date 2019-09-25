#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from global_dict.w_global import gbl_get_value


def write_XY(temp_x, data):

    # n_slice, n_pixel, n_pixel, slice_x
    slice_x = gbl_get_value("slice_x")
    n_pixel = gbl_get_value("img_shape")[0]
    n_slice = gbl_get_value("img_shape")[2]
    X = np.zeros((n_slice, n_pixel, n_pixel, slice_x))
    Y = np.zeros((n_slice, n_pixel, n_pixel, 1))

    # write X
    if slice_x == 1:
        for idx in range(n_slice):
            X[idx, :, :, 0] = temp_x[:, :, idx]
            Y[idx, :, :, 0] = data[:, :, idx]

    if slice_x == 3:
        for idx in range(n_slice):
            idx_0 = idx - 1 if idx > 0 else 0
            idx_1 = idx
            idx_2 = idx + 1 if idx < n_slice - 1 else n_slice - 1
            X[idx, :, :, 0] = temp_x[:, :, idx_0]
            X[idx, :, :, 1] = temp_x[:, :, idx_1]
            X[idx, :, :, 2] = temp_x[:, :, idx_2]
            Y[idx, :, :, 0] = data[:, :, idx]

    return X, Y