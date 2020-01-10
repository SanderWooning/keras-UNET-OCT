import scipy.io as sio
import numpy as np, sys, os
import matplotlib.pyplot as plt


"""
Path_layers is the location where all the .mat files are stored
Save_dir is the location where 2 directories need to be made called im/ and l1/, this is where the images and layers are stored
"""
path_layers = "DirectoryToMatFiles"
save_dir = "SaveDirectory"

my_dpi = 200

# 1. Iterate through all the dataset
for dirName, subdirList, fileList in sorted(os.walk(path_layers)):
    for filename in fileList:
        if ".mat" in filename.lower():
            mat_contents = sio.loadmat(path_layers + filename)

            images = mat_contents['images']
            layers1 = mat_contents['layerMaps']  # for DUKE

            x, y, nimages = images.shape
            step = 4
            ini = int(y / step)
            fin = int(ini * (step - 1))

            thr = fin - ini - 1  # min value to be considered to be a useful segmentation

            init_images = 30  # first used image (the beginning and end of the stack are not segmented), change to 0 to use all the volume
            step_images = 1  # change to 1 to use all the images in the interval
            end_images = 71  # last used image (the beginning and end of the stack are not segmented), change to 100 to use all the volume

            for i in range(init_images, end_images, step_images):

                curr_im = images[:, ini:fin, i]

                curr_l1_0 = layers1[i, ini:fin, 0]
                curr_l1_1 = layers1[i, ini:fin, 1]
                curr_l1_2 = layers1[i, ini:fin, 2]

                cn0 = np.count_nonzero(~np.isnan(curr_l1_0))
                cn1 = np.count_nonzero(~np.isnan(curr_l1_1))
                cn2 = np.count_nonzero(~np.isnan(curr_l1_2))

                flag = ((cn0 > thr) and (cn1 > thr) and (cn2 > thr))

                if flag:
                    """
                    Renaming of the files as the following format, example: 100843.tif
                    Where 1008 is the subject number
                    Where 43 is the slice of the total sequence of the mat file
                    """

                    name = filename[:-4].split("_")[-1]
                    name = name[1:]

                    name_curr_im = save_dir + "im/" + "1" + name + str(i) + ".tif"
                    name_curr_l1 = save_dir + "l1/" + "1" + name + str(i) + ".tif"

                    # 2. Print each B-scan to file
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(y / my_dpi, x / my_dpi)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(curr_im, aspect='auto', cmap='gray')
                    fig.savefig(name_curr_im, dpi=my_dpi, pad_inches=0)

                    # 3. Print each associated layer to file (as white over black, creating a mask)
                    black = np.zeros([x, fin - ini], dtype=curr_im.dtype)
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(y / my_dpi, x / my_dpi)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(black, aspect='auto', cmap='gray')
                    ax.plot(curr_l1_0, 'w', aa=False   , linewidth=0.25)
                    ax.plot(curr_l1_1, 'w', aa=False, linewidth=0.25)
                    ax.plot(curr_l1_2, 'w', aa=False, linewidth=0.25)
                    fig.savefig(name_curr_l1, dpi=my_dpi, pad_inches=0)

                    plt.close('all')

