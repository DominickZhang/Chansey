#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import glob
import nibabel as nib
import numpy as np
from keras import backend as K
from process.small_toys import MaxMinNorm
from global_dict.w_global import gbl_get_value


def mean_squared_error_1e6(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=-1)

    # Energy conservation
    # reg_term = K.square(K.sum(y_pred) - K.sum(y_true))
    # loss += reg_term
    return loss


def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    return (20 - 10 * K.log(mse) / K.log(10.0))


def predict_MRCT(model, test_path, tag=''):
    path_X = glob.glob(test_path + '*Align*.nii*')[-1]
    file_X = nib.load(path_X)
    data_X = file_X.get_fdata()

    path_Y = glob.glob(test_path + '*CT*.nii*')[-1]
    file_Y = nib.load(path_Y)
    data_Y = file_Y.get_fdata()

    # MaxMin-norm
    data_input, X_max, X_min = MaxMinNorm(data_X, keep_para=True)

    print("data shape", data_input.shape)

    n_pixel = data_X.shape[0]
    n_slice = data_X.shape[2]
    slice_x = gbl_get_value("slice_x")
    model_id = gbl_get_value("model_id")
    dir_syn = gbl_get_value('dir_syn')

    # print("y_hat shape: ", y_hat.shape)

    X = np.zeros((1, n_pixel, n_pixel, slice_x))
    y_hat = np.zeros((n_pixel, n_pixel, n_slice))
    if slice_x == 1:
        for idx in range(n_slice):
            X[0, :, :, 0] = data_input[:, :, idx]
            y_hat[:, :, idx] = np.squeeze(model.predict(X))

    if slice_x == 3:
        for idx in range(n_slice):
            idx_0 = idx-1 if idx > 0 else 0
            idx_1 = idx
            idx_2 = idx+1 if idx < n_slice-1 else n_slice - 1
            X[0, :, :, 0] = data_input[:, :, idx_0]
            X[0, :, :, 1] = data_input[:, :, idx_1]
            X[0, :, :, 2] = data_input[:, :, idx_2]
            y_hat[:, :, idx] = np.squeeze(model.predict(X))

    print("Output:")
    print("Mean:", np.mean(y_hat))
    print("STD:", np.std(y_hat))
    print("Max:", np.amax(y_hat))
    print("Min:", np.amax(y_hat))


    # norm y_hat
    y_hat_norm = MaxMinNorm(y_hat)

    # restore norm
    y_hat_norm *= (X_max-X_min)
    y_hat_norm += X_min
    dif = y_hat_norm - data_Y

    # save nifty file
    affine = file_Y.affine
    header = file_Y.header
    nii_file = nib.Nifti1Image(y_hat_norm, affine, header)
    nib.save(nii_file, dir_syn + 'syn_'+model_id+'_'+tag+'.nii')
    print(dir_syn + 'syn_'+model_id+'.nii')

    # save difference

    dif_file = nib.Nifti1Image(dif, affine, header)
    nib.save(dif_file, dir_syn + 'dif_' + model_id+'_'+tag+'.nii')
