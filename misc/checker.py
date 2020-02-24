from psd_tools import PSDImage
import psd_tools
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os


# Check the contents of the object (the names and shapes of the different layers)
path_layers = "D:/Erasmus/Rotterdam/oct_layers/"
save_dir = "D:/Erasmus/Rotterdam/save/"

my_dpi = 200

for dirName, subdirList, fileList in sorted(os.walk(path_layers)):
    for filename in fileList:
        if ".psd" in filename.lower():
            psd = PSDImage.open(path_layers + filename)

            list(psd.descendants())

            # Separate each layer (and original B-scan)
            img_bscan = psd[0]
            img_bm = psd[1]
            img_rpe = psd[2]
            img_ilm = psd[3]

            # Obtain the bounding boxes for each layer
            bbox_bscan = img_bscan.bbox
            bbox_bm = img_bm.bbox
            bbox_rpe = img_rpe.bbox
            bbox_ilm = img_ilm.bbox

            # Obtain also the numpy arrays, to be able to manipulate the images easily
            # We select the position 0 because PIL generates a 4-channel image (RGB + alpha)
            # but in truth we work with grayscale, so channels 0-2 contain the same information
            arr_bscan = np.array(img_bscan.as_PIL())[:,:,0]
            arr_bm = np.array(img_bm.as_PIL())[:,:,0]
            arr_rpe = np.array(img_rpe.as_PIL())[:,:,0]
            arr_ilm = np.array(img_ilm.as_PIL())[:,:,0]

            # A bounding box has the shape: (start_x, start_y, end_x, end_y)
            # Create a base image with the same size than the B-scan
            shape_x = bbox_bscan[2]-bbox_bscan[0]
            shape_y = bbox_bscan[3]-bbox_bscan[1]
            black_img = np.zeros((shape_y, shape_x), np.uint8)

            # Now, use that image as basis to "stick" each layer
            # We use the subtraction because the layers are stored as black over white
            # (white/positive class is the background), but it can be easily changed
            #plt.imshow(arr_bscan, cmap="gray")
            #plt.show()

            mask_bm = black_img.copy()
            mask_bm[bbox_bm[1]:bbox_bm[3],bbox_bm[0]:bbox_bm[2]] = 255-arr_bm
            #plt.figure(1)
            #plt.imshow(mask_bm)

            mask_rpe = black_img.copy()
            mask_rpe[bbox_rpe[1]:bbox_rpe[3],bbox_rpe[0]:bbox_rpe[2]] = 255-arr_rpe
            #plt.figure(2)
            #plt.imshow(mask_rpe)

            mask_ilm = black_img.copy()
            mask_ilm[bbox_ilm[1]:bbox_ilm[3],bbox_ilm[0]:bbox_ilm[2]] = 255-arr_ilm
            #plt.figure(3)
            #plt.imshow(mask_ilm)
            #plt.show()

            mask_total = mask_bm + mask_ilm + mask_rpe

            name_curr_im = save_dir + "image/" + filename + ".tif"
            name_curr_layer = save_dir + "label/" + filename + ".tif"

            fig = plt.figure(frameon=False)
            fig.set_size_inches(shape_x / my_dpi, shape_y / my_dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(arr_bscan, aspect='auto', cmap='gray')
            fig.savefig(name_curr_im, dpi=my_dpi, pad_inches=0)

            fig = plt.figure(frameon=False)
            fig.set_size_inches(shape_x / my_dpi, shape_y / my_dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(mask_total, aspect='auto', cmap='gray')
            fig.savefig(name_curr_layer, dpi=my_dpi, pad_inches=0)