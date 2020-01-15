import keras
import h5py
import numpy as np
from keras.models import load_model
import PIL
from skimage.io import imread, imshow, imread_collection, concatenate_images
import imageio
import matplotlib
import os, os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import cv2 as cv
import numpy as np
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from tqdm import tqdm
from losses2 import *
from skimage import measure
from skimage.morphology import medial_axis, skeletonize
from oct_mincostpath import *
from skimage.feature.texture import greycoprops, greycomatrix

from scipy.signal import find_peaks
from PIL import Image

matplotlib.use('Agg')
plt.style.use('dark_background')

model = load_model("/nfs/home1/swooning/model_thinlabel.hdf5",
                   custom_objects={'generalised_dice_loss_2d': generalised_dice_loss_2d,
                                   'dice_coef': dice_coef}
                   )


seed = 453

DIR = "/nfs/home1/swooning/Small_testing_set/image/image"
file_amount = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
batchsize = 16

''' 
RemovedSmall removed small object which are interlinked together.
Change min_size to set the threshold

Input: Image (2D Numpy array) with small artifacts
Output: Image (2D Numpy array) without artifacts

'''


def RemoveSmall(image, i):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    min_size = 900
    img2 = np.zeros((output.shape))
    for x in range(0, nb_components):
        if sizes[x] >= min_size:
            img2[output == x + 1] = 1
    return (img2)


def GapFill(image, i):
    kernel = np.ones((15, 10), np.uint8)
    d_im = cv.dilate(image, kernel, iterations=1)
    e_im = cv.erode(d_im, kernel, iterations=1)
    return (e_im)


def Skeleton(image):
    skel, distance = medial_axis(image, return_distance=True)
    dist_on_skel = distance * skel
    skeleton_lee = skeletonize(image)
    return (skeleton_lee)


def Contouring(image):
    ret, thresh = cv.threshold(image, 127, 255, 0)
    image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    return (img)


# def LineTrace(imagearray):
#     top_line_array = np.zeros((512, 512), dtype=np.uint16)
#     top_oned_array = np.zeros((512), dtype=np.uint16)
#
#     bottom_line_array = np.zeros((512, 512), dtype=np.uint16)
#     bottom_oned_array = np.zeros((512), dtype=np.uint16)
#
#
#     twistedarray = np.rot90(np.rot90(imagearray))
#
#     num = 0
#
#     for j in range(0, 512):
#         for i in range(0, 512):
#             if imagearray[i, j] == 1:
#                 num += 1
#                 if num == 3:
#                     top_line_array[i, j] = 1
#                     top_oned_array[j] = 512 - i
#                     num = 0
#                     break
#
#     for j in range(0, 512):
#         for i in range(0, 512):
#             if twistedarray[i, j] == 1:
#                 num += 1
#                 if num == 3:
#                     bottom_line_array[i, j] = 1
#                     num = 0
#                     break
#
#     backtwistedarray = np.rot90(np.rot90(bottom_line_array))
#
#     for j in range(0, 512):
#         for i in range(0, 512):
#             if backtwistedarray[i, j] == 1:
#                 bottom_oned_array[j] = 512 - i
#                 break
#
#     bottom_right_image = np.rot90(np.rot90(bottom_line_array))
#     double_mask = np.dstack((top_line_array, bottom_right_image))
#     return (double_mask, top_oned_array, bottom_oned_array )

def Costfunction(image):
    print("Preparing cost images for ")

    # Turn it into a floating point image
    image = image.astype(np.float)
    # image = np.flip(image,0)
    # image = np.flip(image,1)
    # plt.imshow(image, cmap="gray")
    # Normalize the image
    # image = normalize_image(image)

    # COST IMAGE COMPONENTS...

    # Normalize the image
    # image = normalize_image(image)

    # COST IMAGE COMPONENTS...

    image_blur = cv.GaussianBlur(image, (3, 3), 2, 2)
    image_blur_norm = normalize_image(image_blur)

    # Derivative in y (for RPE cost images)
    image_dy = cv.Sobel(image, ddepth=cv.CV_64F,
                        dx=0, dy=1, ksize=3, borderType=cv.BORDER_DEFAULT)
    image_dy_norm = normalize_image(image_dy)

    # Smooth slope in y (for prefering top or bottom contours)
    y_grad_factor_ilm = 1
    y_grad_factor_rpe = 0.2
    smooth_y_grad = np.broadcast_to(np.arange(0.0, image.shape[1])[:, None], image.shape)
    smooth_y_grad_ilm = y_grad_factor_ilm / image.shape[1] * smooth_y_grad
    smooth_y_grad_rpe = y_grad_factor_rpe / image.shape[1] * smooth_y_grad

    # Create initial cost images, top/bottom are for RPE
    cost_image_ilm = 1.0 - image_blur_norm + smooth_y_grad_ilm
    cost_image_top = 1.0 - image_dy_norm + smooth_y_grad_rpe
    cost_image_bottom = image_dy_norm + y_grad_factor_rpe - smooth_y_grad_rpe

    # Parameters
    step = 2

    # Trace ILM
    print("Trace ILM")
    cumcostimage_ilm = min_cost_path_octa(cost_image_ilm, step)
    path_ilm = trace_back_2D(cumcostimage_ilm, step)

    # Find bottom of ILM to crop for next stage
    # bottom_ilm = max(path_ilm, key = lambda t: t[1])[1] + 5
    bottom_ilm = 0
    print("bottom_ilm = {:g}".format(bottom_ilm))

    xdim = cost_image_ilm.shape[1]
    ydim = cost_image_ilm.shape[0]
    for (x, y) in path_ilm:
        miny = max(0, y - 6)
        maxy = min(ydim, y + 6)
        cost_image_top[miny:maxy, x] = 1
        cost_image_bottom[miny:maxy, x] = 1

    # print("Trace RPE-DE")
    # cumcostimage_top = min_cost_path_octa(cost_image_top, step)
    # path_top = trace_back_2D(cumcostimage_top, step)

    # print("Trace RPE-OBM")
    # cumcostimage_bottom = min_cost_path_octa(cost_image_bottom, step)
    # path_bottom = trace_back_2D(cumcostimage_bottom, step)

    # Trace RPE-DE
    print("Trace RPE-DE cropped")
    cost_image_top_cropped = cost_image_top[bottom_ilm:, :]
    cumcostimage_top = min_cost_path_octa(cost_image_top_cropped, step)
    path_top = trace_back_2D(cumcostimage_top, step)

    # Trace RPE-OBM (could be made faster, by cropping based on the
    # RPE-DE results, similar to below for the 2-contour version.)
    # print("Trace RPE-OBM cropped")
    # cost_image_bottom_cropped = cost_image_bottom[bottom_ilm:, :]
    # cumcostimage_bottom = min_cost_path_octa(cost_image_bottom_cropped, step)
    # path_bottom s trace_back_2D(cumcostimage_bottom, step)

    # Trace RPE-OBs, with rpe-de distance
    print("Trace RPE-OBM with RPE-DE distance constrained")
    rpe_dist = 7
    large_cost = 100
    # IMPORTANT: we do a full copy of the cropped image. This is to
    # prevent changing the original cost imgae in when adding the large_cost
    # to pixels too close to the RPE-DE.
    # If the image is sliced without a copy, updating the cropped imgae
    # will also change the origianl image, affecting e.g. the 2-contour
    # minimum cost path below.
    cost_image_bottom_cropped_dist = np.copy(cost_image_bottom[bottom_ilm:, :])
    for (x, y) in path_top:
        cost_image_bottom_cropped_dist[0:y + rpe_dist, x] = large_cost
    cumcostimage_bottom_dist = min_cost_path_octa(cost_image_bottom_cropped_dist, step)
    path_bottom_dist = trace_back_2D(cumcostimage_bottom_dist, step)

    # # Trace both simultaneous
    # # First crop the images, to make it faster...
    # print("Trace RPE-DE and RPE-OBM")
    # top_rpe_de = min(path_top, key=lambda t: t[1])[1]
    # bottom_rpe_de = max(path_top, key=lambda t: t[1])[1]
    # top_rpe_de = top_rpe_de + bottom_ilm - 5
    # bottom_rpe_de = bottom_rpe_de + bottom_ilm + 50
    # print("top_rpe_de = {:g}".format(top_rpe_de))
    # print("bottom_rpe_de = {:g}".format(bottom_rpe_de))
    #
    # cost_image_top_cropped_2 = cost_image_top[top_rpe_de:bottom_rpe_de, :]
    # cost_image_bottom_cropped_2 = cost_image_bottom[top_rpe_de:bottom_rpe_de, :]

    # cum_cost_image_2 = min_cost_2_path_octa(cost_image_top_cropped_2,
    #                                         cost_image_bottom_cropped_2,
    #                                         step, step)
    # path_top_2, path_bottom_2 = trace_back_3D(cum_cost_image_2, step)

    # Write all results to a file (should depend on filename!)
    empty_array = np.zeros((512, 512), dtype=np.uint8)

    for p in range(0, len(path_top)):
        # for p in range(0, len(path_ilm)):
        ILM_numbers_x = path_ilm[p][0]
        ILM_numbers_y = path_ilm[p][1]
        # print(ILM_numbers_x, ILM_numbers_y)
        RPE_numbers_x = (path_top[p][0])
        RPE_numbers_y = (path_top[p][1])
        # print(RPE_numbers_x, RPE_numbers_y)
        OBM_numbers_x = (path_bottom_dist[p][0])
        OBM_numbers_y = (path_bottom_dist[p][1])
        # print(OBM_numbers_x, OBM_numbers_y)

        empty_array[ILM_numbers_x, ILM_numbers_y] = 2
        empty_array[RPE_numbers_x, RPE_numbers_y] = 3
        empty_array[OBM_numbers_x, OBM_numbers_y] = 4

    empty_array = np.rot90(empty_array, -1)
    empty_array = np.flip(empty_array, 1)

    return (empty_array, path_top, path_bottom_dist)



def HistoDrusen(top_oned_array, bottom_oned_array, imagearray, i):
    image_drusen = np.full((512, 512), 0, dtype=np.int16)
    avarage_array = np.zeros((512), dtype=np.int8)

    for p in range(0, 512):
        bottom = (bottom_oned_array[p][1])
        top = (top_oned_array[p][1])
        avarage = (bottom - top)
        avarage_array[p] = (bottom - top)
        for y in range(0, avarage):
            differ = top + y
            image_drusen[p, differ] = 1

    image_drusen = (np.rot90(image_drusen))
    imagearray = np.rot90(imagearray, 2)
    # image_drusen = np.ma.masked_where(image_drusen == 300, image_drusen)
    drusenarray = image_drusen * imagearray
    gradient_drusen = avarage_array * image_drusen

    drusenarray = drusenarray.astype(np.uint8)
    gradient_drusen = gradient_drusen.astype(np.uint8)
    gradient_drusen = np.rot90(gradient_drusen, 2)
    gradient_drusen = np.ma.masked_where(gradient_drusen == 0, gradient_drusen)

    histogram_plot = cv.calcHist([drusenarray], [0], None, [256], [1, 256])

    # plt.savefig("Histogram" + str(i) + ".png")
    # plt.close()
    return (drusenarray, gradient_drusen, avarage_array,image_drusen, histogram_plot)


def DrusenFinder(top_oned_array, bottom_oned_array, drusenarray):
    hight_avarage = np.zeros((512, 512), dtype=np.int16)

    for q in range(0, 512):
        bottom = (bottom_oned_array[q][1])
        top = (top_oned_array[q][1])
        avarage = (top - bottom)
        for y in range(0, avarage):
            differ = bottom + y
            hight_avarage[p, differ] = 1
    hight_avarage = (np.rot90(image_drusen))
    # if avarage <= 8:
    #     for y in range(0, avarage):
    #         differ = bottom + y
    #         hight_avarage[x, differ] = 4
    # if avarage >= 9 and avarage <= 11:
    #     for y in range(0, avarage):
    #         differ = bottom + y
    #         hight_avarage[x, differ] = 3
    # if avarage >= 12 and avarage <= 15:
    #     for y in range(0, avarage):
    #         differ = bottom + y
    #         hight_avarage[x, differ] = 2
    # if avarage >= 16:
    #     for y in range(0, avarage):
    #         differ = bottom + y
    #         hight_avarage[x, differ] = 1

    hight_avarage = np.rot90(hight_avarage)
    hight_avarage = hight_avarage.astype(np.uint8)
    return (hight_avarage)


def Maskgenerator(generatorfile_image, generatorfile_GT, Imgoverlay=True, GToverlay=False):
    real_image_stack = generatorfile_image[0]
    real_GT_stack = generatorfile_GT[0]
    for i in range(0, file_amount):
        real_image = real_image_stack[i, :, :, :]
        image = real_image[:, :, 0]
        normal_image = real_image[:, :, 0]
        # plt.imshow(image, interpolation='none', cmap='gray')

        fig = plt.figure(figsize=(10,8), dpi=300)
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_xlim([0, 512])
        ax1.set_ylim([512, 0])
        ax2 = fig.add_subplot(1, 2, 2)

        if Imgoverlay == True:
            ax1.imshow(image, interpolation='none', cmap='gray')



        predictions = model.predict(real_image_stack)
        two = predictions[i, :, :, 0]
        two = np.where(two > 0.4, 1, 0)
        two = two.astype(np.uint8)
        # plt.imshow(two)
        # plt.savefig("Predict_Only"+str(i)+".png")

        GapImage = GapFill(two, i)

        RemovedImage = RemoveSmall(GapImage, i)
        # labeled_mask = measure.label(RemovedImage, connectivity=2)
        # ilm_mask = (labeled_mask==1)
        #
        # Pre_seperated_mask = (labeled_mask==2).astype(np.uint8)
        # Pre_seperated_mask = Pre_seperated_mask.astype(np.float32)

        empty_array, path_top, path_bottom_dist = Costfunction(image=RemovedImage)
        drusen_array, gradient_drusen, avarage_array, image_drusen, histogram_plot = HistoDrusen(top_oned_array=path_top,
                                                                   bottom_oned_array=path_bottom_dist,
                                                                   imagearray=normal_image, i=i)


        ax2.set_xlim([1, 256])
        ax2.set_ylim([0, 500])
        ax2.plot(histogram_plot)

        x = histogram_plot[:, 0]
        print(x)
        peaks, properties = find_peaks(x, prominence=1, width=3)
        ax2.plot(peaks, x[peaks], "x")
        ax2.vlines(x=peaks, ymin=x[peaks] - properties["prominences"], ymax=x[peaks], color="C1")
        ax2.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color="C1")

        # drusenfinder = DrusenFinder(top_oned_array=top_oned_array, bottom_oned_array=bottom_oned_array, drusenarray=)
        # drusenfinder = np.ma.masked_where(drusenfinder == 0, drusenfinder)
        # drusenarray, gradient_drusen, avarage_array = HistoDrusen(top_oned_array=top_oned_array, bottom_oned_array=bottom_oned_array, imagearray=image, i=i)

        # ILM_mask = (labeled_mask==1).astype(np.uint8)
        # ILM_mask = np.ma.masked_where(ILM_mask == 0, ILM_mask)
        # OBM_mask = OBM_RPE_seperated_mask[:,:,0]
        # OBM_mask = np.ma.masked_where(OBM_mask == 0, OBM_mask)
        #
        # #
        # RPE_mask = OBM_RPE_seperated_mask[:,:,1]
        # RPE_mask = np.ma.masked_where(RPE_mask == 0, RPE_mask)

        if GToverlay == True:
            GT = real_GT_stack[i, :, :, 0]
            GT = GT.astype(np.uint8)
            data_mask = np.ma.masked_where(GT == 0, GT)
            plt.imshow(data_mask, interpolation='none', cmap='brg', alpha=0.5, vmin=0)

        total_drusen = 0
        for m in range(0,len(avarage_array)):
            col_avarage = avarage_array[m]
            if col_avarage >= 0:
                total_drusen = total_drusen + col_avarage

  
        mean = np.zeros((file_amount),dtype=np.float32)
        std = np.zeros((file_amount),dtype=np.float32)
        var = np.zeros((file_amount),dtype=np.float32)
        contrast= np.zeros((file_amount),dtype=np.float32)
        homogeneity = np.zeros((file_amount),dtype=np.float32)
        energy = np.zeros((file_amount),dtype=np.float32)
        
        mean[i] = np.mean(histogram_plot, dtype=np.float32)
        std[i] = np.std(histogram_plot, dtype=np.float32)
        var[i] = np.var(histogram_plot, dtype=np.float32)
        drusen_array = np.ma.masked_where(drusen_array == 0, drusen_array)
  
        distances = [1,5,10,15,20]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ["contrast", "homogeneity", "energy"]

        glcm = greycomatrix(drusen_array,
                            distances = distances,
                            angles = angles,
                            levels = 256,
                            symmetric = True,
                            normed = True)
        contrast = greycoprops(glcm, prop = "contrast")
        homogeneity = greycoprops(glcm, prop = "homogeneity")
        energy = greycoprops(glcm, prop = "energy")
        correlation = greycoprops(glcm, prop = "correlation")
        dissimilarity = greycoprops(glcm, prop = "dissimilarity") 

        '''
        Average Drusen Height Calc
        '''
        avarage_rpeheight = np.average(avarage_array)

        #print(texture) 
        #print(histogram_plot)

        # # topvalue = np.amax(avarage_array)
        #
        empty_array = np.ma.masked_where(empty_array == 0, empty_array)
        ax1.imshow(empty_array, interpolation='none', alpha=0.8, cmap="brg")
        # ilm_mask = np.ma.masked_where(ilm_mask == 0, ilm_mask)
        # plt.imshow(ilm_mask, interpolation='none', alpha=0.8, cmap="brg")
        ax1.imshow(gradient_drusen, interpolation='none', alpha=0.5, vmin=8, vmax=28, cmap="RdYlGn_r")
        # plt.imshow(OBM_mask, interpolation='none', alpha=0.8, cmap='gist_rainbow', vmax=1)
        # plt.imshow(RPE_mask, interpolation='none', alpha=0.8, cmap="rainbow", vmax=1)

        ax1.text(10.0, 600.0, "DrusenPixs: " + str(total_drusen),
                verticalalignment='bottom', horizontalalignment='left',
                color='white', fontsize=8)
        ax1.text(10.0, 630.0, "mean: " + str(mean[i]),
                verticalalignment='bottom', horizontalalignment='left',
                color='white', fontsize=8)
        ax1.text(10.0, 660.0, "STD: " + str(std[i]),
                verticalalignment='bottom', horizontalalignment='left',
                color='white', fontsize=8)
        ax1.text(10.0, 690.0, "Var: " + str(var[i]),
                 verticalalignment='bottom', horizontalalignment='left',
                 color='white', fontsize=8)
        ax1.text(10.0, 720.0, "RPE-OBM " + str(avarage_rpeheight),
                 verticalalignment='bottom', horizontalalignment='left',
                 color='white', fontsize=8)

        ax1.text(180.0, 600.0, "homogeneity " + str(np.average(homogeneity)),
                verticalalignment='bottom', horizontalalignment='left',
                color='white', fontsize=8)
        ax1.text(180.0, 630.0, "energy: " + str(np.average(energy)),
                verticalalignment='bottom', horizontalalignment='left',
                color='white', fontsize=8)
        ax1.text(180.0, 660.0, "contrast: " + str(np.average(contrast)),
                 verticalalignment='bottom', horizontalalignment='left',
                 color='white', fontsize=8)   
        ax1.text(180.0, 690.0, "correlation: " + str(np.average(correlation)),
                 verticalalignment='bottom', horizontalalignment='left',
                 color='white', fontsize=8)                    
        ax1.text(180.0, 720.0, "dissimilairtiy: " + str(np.average(dissimilarity)),
                 verticalalignment='bottom', horizontalalignment='left',
                 color='white', fontsize=8)                                    
        plt.savefig("GDL_CHECKKING_" + str(i) + "_Final-image.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    # np.save("drusenarray_for_danilo.npy", empty_3d_array)




def Masterswitch(evaluate=False, predict=False, Imgoverlay=True, GToverlay=True):
    if evaluate == True and predict == False:
        evaluate_img_generator = ImageDataGenerator()
        evaluate_img_two_generator = evaluate_img_generator.flow_from_directory(
            "/nfs/home1/swooning/ThinData/test/image",
            target_size=(512, 512),
            color_mode="grayscale",
            batch_size=16,
            class_mode=None,
            shuffle=True,
            seed=seed)

        evaluate_mask_generator = ImageDataGenerator()
        evaluate_mask_two_generator = evaluate_mask_generator.flow_from_directory(
            "/nfs/home1/swooning/ThinData/test/label",
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

    if evaluate == False and predict == True:
        predict_image_generator = ImageDataGenerator()
        predict_two_generator = predict_image_generator.flow_from_directory(
            "/nfs/home1/swooning/Small_testing_set/image",
            target_size=(512, 512),
            color_mode="grayscale",
            batch_size=200,
            class_mode=None,
            shuffle=False,
            #save_to_dir="predict/image",
            seed=seed)

        predict_mask_generator = ImageDataGenerator()
        predict_mask_two_generator = predict_mask_generator.flow_from_directory(
            "/nfs/home1/swooning/Small_testing_set/label",
            target_size=(512, 512),
            color_mode="grayscale",
            batch_size=16,
            class_mode=None,
            shuffle=False,
            #save_to_dir="predict",
            seed=seed)
        Maskgenerator(predict_two_generator, predict_mask_two_generator, Imgoverlay=True, GToverlay=False)

    if evaluate == True and predict == True:
        evaluate_img_generator = ImageDataGenerator()
        evaluate_img_two_generator = evaluate_img_generator.flow_from_directory(
            "/nfs/home1/swooning/Small_testing_set/image",
            target_size=(512, 512),
            color_mode="grayscale",
            batch_size=16,
            class_mode=None,
            shuffle=True,
            seed=seed)

        evaluate_mask_generator = ImageDataGenerator()
        evaluate_mask_two_generator = evaluate_mask_generator.flow_from_directory(
            "/nfs/home1/swooning/Small_testing_set/label",
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


Masterswitch(evaluate=False, predict=True)