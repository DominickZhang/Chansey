#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import glob
import datetime
import argparse
import numpy as np
import nibabel as nib
from global_dict.w_global import gbl_set_value, gbl_get_value, gbl_save_value
from process.MRCT_load import write_XY
from process.small_toys import MaxMinNorm
from model.w_train import train_a_unet
from predict.MRCT_predict import predict_MRCT

np.random.seed(591)


def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--train_case', metavar='', type=int, default=1,
                        help='The training dataset case(1)<int>[1,2,3,4]')
    parser.add_argument('--test_case', metavar='', type=int, default=2,
                        help='The testing dataset case(2)<int>[1,2,3,4]')

    parser.add_argument('--slice_x', metavar='', type=int, default="1",
                        help='Slices of input(1)<int>[1/3]')
    parser.add_argument('--id', metavar='', type=str, default="chansey",
                        help='ID of the current model.(eeVee)<str>')

    parser.add_argument('--epoch', metavar='', type=int, default=240,
                        help='Number of epoches of training(300)<int>')
    parser.add_argument('--n_filter', metavar='', type=int, default=64,
                        help='The initial filter number(64)<int>')
    parser.add_argument('--depth', metavar='', type=int, default=3,
                        help='The depth of U-Net(4)<int>')
    parser.add_argument('--batch_size', metavar='', type=int, default=5,
                        help='The batch_size of training(10)<int>')

    args = parser.parse_args()

    train_case = args.train_case
    test_case = args.test_case

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = args.id + time_stamp
    gbl_set_value("depth", args.depth)
    gbl_set_value("dir_syn", './walmart/')
    gbl_set_value("dir_model", './walmart/')
    gbl_set_value("model_id", model_id)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("n_filter", args.n_filter)
    gbl_set_value("depth", args.depth)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("slice_x", args.slice_x)

    # Load data
    train_path = './data/MRCT/Case'+str(train_case)+'/'
    test_path = './data/MRCT/Case'+str(test_case)+'/'

    path_X = glob.glob(train_path+'*Align*.nii')[-1]
    path_Y = glob.glob(train_path+'*CT*.nii')[-1]

    file_X = nib.load(path_X)
    file_Y = nib.load(path_Y)

    data_X = file_X.get_fdata()
    data_Y = file_Y.get_fdata()

    #MaxMin-norm
    data_X_norm = MaxMinNorm(data_X)
    data_Y_norm = MaxMinNorm(data_Y)

    gbl_set_value("img_shape", data_X.shape)
    X, Y = write_XY(data_X_norm, data_Y_norm)

    print(X.shape, Y.shape)
    print("Loading Completed!")

    model = train_a_unet(X, Y)
    print("Training Completed!")

    predict_MRCT(model, test_path)
    print("Predicting Completed!")

if __name__ == "__main__":
    main()
