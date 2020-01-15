# RPE Drusen Complex Segmentation by UNET 

## Summary

This repo is put together for automatically segmenting the RPE-complex with a convolutional neural network based on the [Unet architecture.](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) Giving the following results in different datasets as shown in the figure down below. A network was trained with Keras backend on [the Duke University SD-OCT datatset](http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm). 

![images/predictions.png](images/predictions.png)

---
## Overview

### Data set & Data preperation
For training the model, a partition of [the Duke University SD-OCT datatset](http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm) was used for training. These images were transfered from their .mat format to .tif images for better use in generators.  


