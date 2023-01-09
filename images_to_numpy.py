import numpy as np
import cv2 as cv
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt

data_path = "C:\\Users\\Tom\\Downloads\\HBP\\multimodalplacerecognition_datasets\\whiskeye_guifen_datasets\\real_data_6005_timestamped\\images\\"

image_files = natsorted(glob(data_path + '*.png'))

final_width = 80
final_height = 45

images = np.zeros(shape=(len(image_files), final_height, final_width, 3))

for i, image_file in enumerate(image_files):

    images[i, :, :, :] = cv.resize(cv.imread(image_file), dsize = (final_width, final_height))[:,:,::-1]/255

    print("Processed {}/{}".format(i+1, len(image_files)), end = '\r')

print("\n")

np.save(data_path + 'images.npy', images)

fig, ax = plt.subplots(5,2)

for i, j in enumerate(range(0, 500, 100)):

    ax[i,0].imshow(cv.imread(image_files[j])[:,:,::-1]/255)
    ax[i,0].set_axis_off()
    ax[i,1].imshow(images[j])
    ax[i,1].set_axis_off()

plt.show()