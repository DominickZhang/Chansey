#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import math
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from global_dict.w_global import gbl_get_value
from model.unet import unet
from model.unet import unet
from skimage.util import random_noise
SEED = 314

def mean_squared_error_1e12(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_squared_error_1e6(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=-1)
    reg_term = K.square(K.sum(y_pred) - K.sum(y_true))
    return loss


def mean_absolute_error_1e6(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


# def psnr(y_true, y_pred):
#     #     return -10.0*K.log(1.0/(K.mean(K.square(y_pred - y_true))))/K.log(10.0)
#     mse = K.mean(K.square(y_pred - y_true))
#     return (20 - 10 * K.log(mse) / K.log(10.0))


def psnr(y_true, y_pred):
    max_pixel = 1.0
    psnr = (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
    return K.minimum(psnr, 100)


def mse1e12_weighted(y_true, y_pred):
    diff = np.dot(K.square(y_pred - y_true), y_pred)
    loss = K.mean(diff, axis=-1)
    return loss


def train_a_unet(X, Y, pretrained_model=None):

    print(X.shape)
    slice_x = gbl_get_value("slice_x")
    n_pixel = X.shape[2]
    n_slice = X.shape[0]
    model_id = gbl_get_value("model_id")
    dir_model = gbl_get_value('dir_model')

    epochs = gbl_get_value("n_epoch")
    n_fliter = gbl_get_value("n_filter")
    depth = gbl_get_value("depth")
    batch_size = gbl_get_value("batch_size")
    optimizer = 'Adam'
    flag_save = True


    # ----------------------------------------------Configurations----------------------------------------------#

    # logs
    log_path = './logs/' + model_id + "/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    tensorboard = TensorBoard(log_dir=log_path, batch_size=batch_size,
                              write_graph=True, write_grads=True,
                              write_images=True)

    # set traininig configurations
    conf = {"image_shape": (n_pixel, n_pixel, slice_x), "out_channel": 1, "filter": n_fliter, "depth": depth,
            "inc_rate": 2, "activation": 'relu', "dropout": True, "batchnorm": True, "maxpool": True,
            "upconv": True, "residual": True, "shuffle": True, "augmentation": True,
            "learning_rate": 1e-5, "decay": 0.0, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999,
            "validation_split": 0.2632, "batch_size": batch_size, "epochs": epochs,
            "loss": "mse1e6", "metric": "mse", "optimizer": optimizer, "model_id": model_id}
    np.save(log_path + model_id + '_info.npy', conf)

    # set augmentation configurations
    conf_a = {"rotation_range": 15, "shear_range": 10,
              "width_shift_range": 0.33, "height_shift_range": 0.33, "zoom_range": 0.33,
              "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'nearest',
              "seed": 314, "batch_size": conf["batch_size"]}
    np.save(log_path + model_id + '__aug.npy', conf_a)

    # checkpoint
    #     check_path= './training_models/' + model_type + '_' + str(LOOCV) + '/'
    #     if not os.path.exists(check_path):
    #         os.makedirs(check_path)
    if flag_save:
        check_path = dir_model+'model_'+model_id+'.hdf5'  # _{epoch:03d}_{val_loss:.4f}
        checkpoint1 = ModelCheckpoint(check_path, monitor='val_psnr',
                                      verbose=1, save_best_only=True, mode='max')
        #     checkpoint2 = ModelCheckpoint(check_path, period=100)
        callbacks_list = [checkpoint1, tensorboard]
    else:
        callbacks_list = [tensorboard]

    # ----------------------------------------------Create Model----------------------------------------------#

    # build up the model
    print(conf)

    if pretrained_model is None:
        model = unet(img_shape=conf["image_shape"], out_ch=conf["out_channel"],
                     start_ch=conf["filter"], depth=conf["depth"],
                     inc_rate=conf["inc_rate"], activation=conf["activation"],
                     dropout=conf["dropout"], batchnorm=conf["batchnorm"],
                     maxpool=conf["maxpool"], upconv=conf["upconv"],
                     residual=conf["residual"])
    else:
        model = pretrained_model

    # Adam optimizer
    # if conf["optimizer"] == 'Adam':
    #     opt = Adam(lr=conf["learning_rate"], decay=conf["decay"],
    #                epsilon=conf["epsilon"], beta_1=conf["beta_1"], beta_2=conf["beta_2"])
    # if conf["loss"] == 'mse1e6':
    #     loss = mean_squared_error_1e6

    loss = mean_squared_error_1e6
    opt = Adam(lr=conf["learning_rate"], decay=conf["decay"],
               epsilon=conf["epsilon"], beta_1=conf["beta_1"], beta_2=conf["beta_2"])

    # load dataset [80, n_pixel, n_pixel, 1]
    # x_val = np.zeros((int(n_slice * 0.3), n_pixel, n_pixel, 1), dtype=np.float32)
    # y_val = np.zeros((int(n_slice * 0.3), n_pixel, n_pixel, 1), dtype=np.float32)
    # x_train = np.zeros((int(n_slice * 0.7) + 1, n_pixel, n_pixel, 1), dtype=np.float32)
    # y_train = np.zeros((int(n_slice * 0.7) + 1, n_pixel, n_pixel, 1), dtype=np.float32)
    #
    # temp_x = X
    # temp_y = Y
    # list_cand = []
    #
    # idx_x = 0
    # while idx_x < int(n_slice * 0.3):
    #     idx_slice = int(np.random.rand() * n_slice)
    #     if not (idx_slice in list_cand):
    #         list_cand.append(idx_slice)
    #         x_val[idx_x, :, :, :] = temp_x[:, :, idx_slice].reshape((1, n_pixel, n_pixel, 1))
    #         y_val[idx_x, :, :, :] = temp_x[:, :, idx_slice].reshape((1, n_pixel, n_pixel, 1))
    #         idx_x += 1
    #
    # idx_x = 0
    # for i in range(n_slice):
    #     if not (i in list_cand):
    #         x_train[idx_x, :, :, :] = temp_x[:, :, i].reshape((1, n_pixel, n_pixel, 1))
    #         y_train[idx_x, :, :, :] = temp_y[:, :, i].reshape((1, n_pixel, n_pixel, 1))
    #         idx_x = idx_x + 1

    X = X.reshape((n_slice, n_pixel, n_pixel, slice_x))
    Y = Y.reshape((n_slice, n_pixel, n_pixel, 1))

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.33, random_state = 42)
    x_train = x_train / np.amax(x_train)
    y_train = y_train / np.amax(y_train)
    x_val = x_val / np.amax(x_val)
    y_val = y_val / np.amax(y_val)

    # ----------------------------------------------Data Generator----------------------------------------------#

    # train data_generator
    data_generator1 = ImageDataGenerator(rotation_range=conf_a["rotation_range"],
                                         shear_range=conf_a["shear_range"],
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)
    data_generator2 = ImageDataGenerator(rotation_range=conf_a["rotation_range"],
                                         shear_range=conf_a["shear_range"],
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)

    # validation data_generator
    data_generator3 = ImageDataGenerator(width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)
    data_generator4 = ImageDataGenerator(width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)

    # set generator
    # data_generator1.fit(x_train, seed=conf_a["seed"])
    # data_generator2.fit(y_train, seed=conf_a["seed"])
    # data_generator3.fit(x_val, seed=conf_a["seed"])
    # data_generator4.fit(y_val, seed=conf_a["seed"])

    aug_dir = ''

    # # save files
    # data_generator1.flow(x_train, seed=conf_a["seed"], save_to_dir=aug_dir, save_prefix='train_x')
    # data_generator2.flow(y_train, seed=conf_a["seed"], save_to_dir=aug_dir, save_prefix='train_y')
    # data_generator3.flow(x_val, seed=conf_a["seed"], save_to_dir=aug_dir, save_prefix='val_x')
    # data_generator4.flow(y_val, seed=conf_a["seed"], save_to_dir=aug_dir, save_prefix='val_y')

    # zip files
    data_generator_t = zip(data_generator1.flow(x=x_train, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='train_x'),
                           data_generator2.flow(x=y_train, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='train_y'))
    data_generator_v = zip(data_generator3.flow(x=x_val, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='val_x'),
                           data_generator4.flow(x=y_val, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='val_y'))

    # ----------------------------------------------Train Model----------------------------------------------#

    # compile
    model.compile(loss=loss, optimizer=opt, metrics=[mean_squared_error_1e6, psnr])

    # train
    model.fit_generator(generator=data_generator_t,
                        steps_per_epoch=int(int(n_slice * 0.7) / conf_a["batch_size"]),  #
                        epochs=conf["epochs"],
                        callbacks=callbacks_list,
                        validation_data=data_generator_v,
                        validation_steps=int(int(n_slice * 0.3) / conf_a["batch_size"]))  #

    return model
