import numpy as np
import cv2 as cv
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt

def images_to_numpy(data_paths, output_paths, final_width, final_height, plot = False) -> None:

    for data_path, output_path in zip(data_paths, output_paths):

        image_files = natsorted(glob(data_path + '*.png'))

        images = np.zeros(shape=(len(image_files), final_height, final_width, 3))

        for i, image_file in enumerate(image_files):

            images[i, :, :, :] = cv.resize(cv.imread(image_file), dsize = (final_width, final_height))[:,:,::-1]/255

            print("Processed {}/{}".format(i+1, len(image_files)), end = '\r')

        print("\n")

        np.save(output_path + 'images.npy', images)

        if plot:

            fig, ax = plt.subplots(5,2)

            for i, j in enumerate(range(0, len(image_files), len(image_files)//5)):

                if i < 5:

                    ax[i,0].imshow(cv.imread(image_files[j])[:,:,::-1]/255)
                    ax[i,0].set_axis_off()
                    ax[i,1].imshow(images[j])
                    ax[i,1].set_axis_off()

            plt.show()

# Original (Guifen-like) arena

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/"

data_paths = [data_root_folder + "training_data_rotation_sets_3000/training_data_centre/images/",
              data_root_folder + "training_data_rotation_sets_3000/training_data_top_left/images/",
              data_root_folder + "training_data_rotation_sets_3000/training_data_top_right/images/",
              data_root_folder + "training_data_rotation_sets_3000/training_data_bottom_left/images/",
              data_root_folder + "training_data_rotation_sets_3000/training_data_bottom_right/images/",
              data_root_folder + "training_data_rotation_sets_3000/training_data_centre_5x/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_bl_tr/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_tr_bl/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_br_tl/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_tl_br/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_lm_rm/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_rm_lm/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_tm_bm/images/",
              data_root_folder + "training_data_translation_sets_3000/training_data_bm_tm/images/"]

output_paths = [data_root_folder + "training_data_rotation_sets_3000/training_data_centre/",
                data_root_folder + "training_data_rotation_sets_3000/training_data_top_left/",
                data_root_folder + "training_data_rotation_sets_3000/training_data_top_right/",
                data_root_folder + "training_data_rotation_sets_3000/training_data_bottom_left/",
                data_root_folder + "training_data_rotation_sets_3000/training_data_bottom_right/",
                data_root_folder + "training_data_rotation_sets_3000/training_data_centre_5x/",
                data_root_folder + "training_data_translation_sets_3000/training_data_bl_tr/",
                data_root_folder + "training_data_translation_sets_3000/training_data_tr_bl/",
                data_root_folder + "training_data_translation_sets_3000/training_data_br_tl/",
                data_root_folder + "training_data_translation_sets_3000/training_data_tl_br/",
                data_root_folder + "training_data_translation_sets_3000/training_data_lm_rm/",
                data_root_folder + "training_data_translation_sets_3000/training_data_rm_lm/",
                data_root_folder + "training_data_translation_sets_3000/training_data_tm_bm/",
                data_root_folder + "training_data_translation_sets_3000/training_data_bm_tm/"]

#images_to_numpy(data_paths, output_paths, final_width = 224, final_height = 224)

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/"

data_paths = [data_root_folder + "real_data_6005_timestamped/images/"]

output_paths = [data_root_folder + "real_data_6005_timestamped/"]

#images_to_numpy(data_paths, output_paths, final_width = 224, final_height = 224)


# Dome arena

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/"

# Missing a regular 'centre' dataset

data_paths = [data_root_folder + "training_dome_rotation_sets_3000/training_dome_top_left/images/",
              data_root_folder + "training_dome_rotation_sets_3000/training_dome_top_right/images/",
              data_root_folder + "training_dome_rotation_sets_3000/training_dome_bottom_left/images/",
              data_root_folder + "training_dome_rotation_sets_3000/training_dome_bottom_right/images/",
              data_root_folder + "training_dome_rotation_sets_3000/training_dome_centre_5x/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_bl_tr/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_tr_bl/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_br_tl/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_tl_br/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_lm_rm/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_rm_lm/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_tm_bm/images/",
              data_root_folder + "training_dome_translation_sets_3000/training_dome_bm_tm/images/"]

output_paths = [data_root_folder + "training_dome_rotation_sets_3000/training_dome_top_left/",
                data_root_folder + "training_dome_rotation_sets_3000/training_dome_top_right/",
                data_root_folder + "training_dome_rotation_sets_3000/training_dome_bottom_left/",
                data_root_folder + "training_dome_rotation_sets_3000/training_dome_bottom_right/",
                data_root_folder + "training_dome_rotation_sets_3000/training_dome_centre_5x/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_bl_tr/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_tr_bl/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_br_tl/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_tl_br/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_lm_rm/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_rm_lm/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_tm_bm/",
                data_root_folder + "training_dome_translation_sets_3000/training_dome_bm_tm/"]

#images_to_numpy(data_paths, output_paths, final_width = 224, final_height = 224)

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/"

data_paths = [data_root_folder + "real_dome_6005_timestamped/images/"]

output_paths = [data_root_folder + "real_dome_6005_timestamped/"]

#images_to_numpy(data_paths, output_paths, final_width = 224, final_height = 224)


# Oracle arena

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/"

# Missing a regular 'centre' dataset

data_paths = [  data_root_folder + "centre/images/",
                data_root_folder + "top_left/images/",
                data_root_folder + "top_right/images/",
                data_root_folder + "bottom_left/images/",
                data_root_folder + "bottom_right/images/",
                data_root_folder + "centre_5x/images/",

                data_root_folder + "bl_tr/images/",
                data_root_folder + "tr_bl/images/",
                data_root_folder + "br_tl/images/",
                data_root_folder + "tl_br/images/",
                data_root_folder + "lm_rm/images/",
                data_root_folder + "rm_lm/images/",
                data_root_folder + "tm_bm/images/",
                data_root_folder + "bm_tm/images/"]

output_paths = [data_root_folder + "centre/",
                data_root_folder + "top_left/",
                data_root_folder + "top_right/",
                data_root_folder + "bottom_left/",
                data_root_folder + "bottom_right/",
                data_root_folder + "centre_5x/",

                data_root_folder + "bl_tr/",
                data_root_folder + "tr_bl/",
                data_root_folder + "br_tl/",
                data_root_folder + "tl_br/",
                data_root_folder + "lm_rm/",
                data_root_folder + "rm_lm/",
                data_root_folder + "tm_bm/",
                data_root_folder + "bm_tm/"]

images_to_numpy(data_paths, output_paths, final_width = 224, final_height = 224, plot = False)

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/"

data_paths = [data_root_folder + "real_rat/images/"]

output_paths = [data_root_folder + "real_rat/"]

images_to_numpy(data_paths, output_paths, final_width = 224, final_height = 224, plot = False)