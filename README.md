# RPE Drusen Complex Segmentation by UNET 

## Summary

This repo is put together for automatically segmenting the RPE-complex with a convolutional neural network based on the [Unet architecture](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). Together with postprocessing-operations, giving the following results in different datasets as shown in the figure 1. A network was trained with Keras backend on [the Duke University SD-OCT datatset](http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm).  Firstly the retinal layers get segmented (ILM, BM and RPE). After postprosessing, the difference between the BM and the RPE is a calculated. Resulting in a Dice-score of 0.979. 

![images/predictions.png](images/predictions.png)

<sub>Figure 1. Image predictions of the model with postprocessing</sub>

---
## Overview

### Data set & Data preperation
A total of 15744 SD-OCT B-scans (Bioptigen SD-OCT, NC, USA) from 269 AMD patients and 115 normal subjects, selected from the Duke dataset, were used in this study for training and cross-validation. The dataset was split on subject level in 80% training, 20% testing. Validation set was later split from the training set in the ImageDataGenerator function of Keras. 

For training the model, a partition of [the Duke University SD-OCT datatset](http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm) was used for training. These images were transfered from their .mat format to .tif images for better use in generators. Only slice 30 to 70 was used, they only had the fully annotated masks. Masks were made by convertining the pixel wide annotated mask to a 3 pixel wide annotated mask which can been seen i. All this preprocessing is done by the code imagemaskfrommat.py. 

|![images/imageduke.png](images/imageduke.png) </br> <sub>Figure 2. Example of an converted OCT image by the .mat file </sub>  	| ![images/maskduke.png](images/maskduke.png) </br> <sub>Figure 3. Example of an 3 pixel wide converted ground thruth mask by the .mat annotation</sub> 	|
|---|---|

### Data Augmentation & Normalization
The Keras ImageDataGenerator has been used to generate batches of augemented images during training by using multi-processing. The geometric & intensity based augementations used are: 

| Augmentation | Type |Value | 
|---|---|
| zoom_range | Geometric Aug | 0.9 , 1.2 |
| width_shift_range <br/> height_shift_range | Geometric Aug| 0.95 , 1.05


### Training & Model 
The U-NET model has been used for segmenting the three retinal layer. Training was done u
