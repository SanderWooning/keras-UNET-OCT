# from model import *
# from data import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import h5py
import numpy as np
from time import time
from keras.models import load_model
import PIL
from skimage.io import imread, imshow, imread_collection, concatenate_images
import imageio

from Model import get_unet

import pathlib
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as K
#data augmentation

from losses2 import *

#https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199


data_gen_args = dict(zoom_range=[0.9, 1.2],
                     width_shift_range=0.05,
                     featurewise_center=True,
                     featurewise_std_normalization=True,
                     height_shift_range=0.05,
                     shear_range=0.1,
                     rotation_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=False,
                     fill_mode='reflect',
                     data_format='channels_last',
                     brightness_range=[0.4, 1.2])

print(data_gen_args)

seed = 1337
batch_size = 16
val_split = 0.1


train_image_path = '/nfs/home1/swooning/ThinData/train/image'
train_label_path = '/nfs/home1/swooning/ThinData/train/label'

steps_epoch = np.floor((len([file.stem for file in pathlib.Path(train_image_path+'/image').iterdir()])/batch_size)*0.8)
steps_val =np.floor((len([file.stem for file in pathlib.Path(train_image_path+'/image').iterdir()])/batch_size)*0.2)

MODEL_NAME = '/nfs/home1/swooning/testing/model_ThinlabelLargeEpoch.hdf5'
print(MODEL_NAME)

train_image_datagen = ImageDataGenerator(rescale=1./255 ,**data_gen_args, validation_split = val_split)
train_label_datagen = ImageDataGenerator(rescale=1./255, **data_gen_args, validation_split = val_split)

train_image_generator = train_image_datagen.flow_from_directory(
       train_image_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset ='training')

train_label_generator = train_label_datagen.flow_from_directory(
       train_label_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset = 'training')

val_image_generator = train_image_datagen.flow_from_directory(
       train_image_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset ='validation')

val_label_generator = train_label_datagen.flow_from_directory(
       train_label_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset = 'validation')

training_generator = zip(train_image_generator, train_label_generator)
validation_generator = zip(val_image_generator, val_label_generator)

input_img = Input((512, 512, 1), name='img')


# Design model
model = get_unet(input_img, n_filters=16, dropout=0.25, batchnorm=False)
# Configuring model for training
model.compile(optimizer=Adam(), loss=generalised_dice_loss_2d, metrics=[dice_coef, "accuracy"])
# model.summary()

callbacks = [
    EarlyStopping(patience=30, monitor='val_dice_coef', verbose=1, mode="max"),
    ReduceLROnPlateau(monitor='val_dice_coef', factor=0.1, patience=5, min_lr=0.0001, verbose=1, mode="max"),
    ModelCheckpoint(MODEL_NAME, monitor='val_dice_coef', save_best_only=True, verbose=1, mode="max")]


# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch = steps_epoch,
                    validation_data = validation_generator,
                    validation_steps = steps_val,
                    epochs=125,
                    verbose=2,
                    use_multiprocessing=True,
                    callbacks=callbacks,
                    workers=1)

evaluate_img_generator = ImageDataGenerator()
evaluate_img_two_generator = evaluate_img_generator.flow_from_directory(
    "/nfs/home1/swooning/Output_Combined/test/image",
    target_size=(512, 512),
    color_mode="grayscale",
    batch_size=16,
    class_mode=None,
    shuffle=True,
    seed=seed)

evaluate_mask_generator = ImageDataGenerator()
evaluate_mask_two_generator = evaluate_mask_generator.flow_from_directory(
    "/nfs/home1/swooning/Output_Combined/test/label",
    target_size=(512, 512),
    color_mode="grayscale",
    batch_size=16,
    class_mode=None,
    shuffle=True,
    seed=seed)

evaluate_generator = zip(evaluate_img_two_generator,
                         evaluate_mask_two_generator)

print('\n# Evaluate on test data')
results = model.evaluate_generator(evaluate_generator, steps=len(evaluate_mask_two_generator),
                                   verbose=1)
print(model.metrics_names, results)
Maskgenerator(evaluate_img_two_generator, evaluate_mask_two_generator, Imgoverlay=True, GToverlay=True)