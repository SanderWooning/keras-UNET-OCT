import keras
import h5py
import numpy as np
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from losses2 import *

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

