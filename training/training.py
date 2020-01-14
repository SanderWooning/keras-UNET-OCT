import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import h5py
import numpy as np
from model import get_unet
from losses import *
import pathlib
from keras.optimizers import Adam

"""
DIRECTORY INDEX
Due to the nature of classes in the Keras Flow From Direcory generator, there needs to be two directories before the images/masks. 
The following directories should be made for the images and their respective masks.
Note: Training and validation data is mixed in one map, this split is later made. 

Data
∟ train & Validation 
|    ∟ image
|    |    ∟ image
|    |       ∟ 100130.tif
|    |         100131.tif
|    |         ...
|    ∟ mask
|        ∟ mask
|            ∟ 100130.tif
|              100131.tif
|               ...
∟ test
    ∟ image
    |    ∟ image
    |       ∟ 100130.tif
    |         100131.tif
    |         ...
    ∟ mask
        ∟ mask
            ∟ 100130.tif
              100131.tif
               ...


"""

"""
All the variable parameters for the data augmentation
"""

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



"""
Generator parameters
"""
seed = 1337
batch_size = 16
val_split = 0.1

train_val_image_path = 'YOURPATH/train/image'
train_val_label_path = 'YOURPATH/train/label'
model_name = 'OCT_DRUSEG_MODEL.hdf5'

print("Data augmentations and normalizations used: ",data_gen_args)
print("Model name: ", model_name)
print("Model parameters: ", batch_size, val_split, )

""""
Automatically calculate the steps per epoch for training and validation
Steps per epoch = (File amount / batchsize) * 1 - validation split 

"""

steps_epoch = np.floor((len([file.stem for file in pathlib.Path(train_val_image_path+'/image').iterdir()])/batch_size)*(1-val_split)
steps_val =np.floor((len([file.stem for file in pathlib.Path(train_val_image_path+'/image').iterdir()])/batch_size)*(val_split))


"""
Apply the rescale to get the pixel-values between 0 and 1
Apply the above mentioned data-augmentation
Splitting the training and validation set
"""

train_image_datagen = ImageDataGenerator(rescale=1./255 ,**data_gen_args, validation_split = val_split)
train_label_datagen = ImageDataGenerator(rescale=1./255, **data_gen_args, validation_split = val_split)

train_image_generator = train_image_datagen.flow_from_directory(
       train_val_image_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset ='training')

train_label_generator = train_label_datagen.flow_from_directory(
       train__val_label_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset = 'training')

val_image_generator = train_image_datagen.flow_from_directory(
       train_val_image_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset ='validation')

val_label_generator = train_label_datagen.flow_from_directory(
       train_val_label_path,
       classes = None,
       class_mode = None,
       color_mode = "grayscale",
       target_size = (512,512),
       batch_size = batch_size,
       seed = seed,
       subset = 'validation')




# Zipping images and masks together
training_generator = zip(train_image_generator, train_label_generator)
validation_generator = zip(val_image_generator, val_label_generator)



# Arguments model
input_img = Input((512, 512, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.25, batchnorm=False)

# Compiling model
model.compile(optimizer=Adam(), loss=generalised_dice_loss_2d(metric= ), metrics=[dice_coef, "accuracy"])
# model.summary()

# Callbacks for training
callbacks = [
    EarlyStopping(patience=30, monitor='val_dice_coef', verbose=1, mode="max"),
    ReduceLROnPlateau(monitor='val_dice_coef', factor=0.1, patience=5, min_lr=0.0001, verbose=1, mode="max"),
    ModelCheckpoint(MODEL_NAME, monitor='val_dice_coef', save_best_only=True, verbose=1, mode="max")]


# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch = steps_epoch,
                    validation_data = validation_generator,
                    validation_steps = steps_val,
                    epochs=100,
                    verbose=2,
                    use_multiprocessing=True,
                    callbacks=callbacks,
                    workers=5) #Lower this value if Keras give an performance issue


