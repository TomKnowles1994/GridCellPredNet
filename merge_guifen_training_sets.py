import numpy as np

def merge_training_data(data_folders, output_folder):

    image_files = [np.load(filename + 'images.npy') for filename in data_folders]

    head_direction_files = [np.load(filename + 'networkOutput_gaussianised.npy') for filename in data_folders]

    output_image_file = np.concatenate(image_files)

    output_head_direction_file = np.concatenate(head_direction_files)

    np.save(output_folder + 'images.npy', output_image_file)

    np.save(output_folder + 'networkOutput_gaussianised.npy', output_head_direction_file)

    for ring in (1, 2, 3):
        for ring_size in (30, 40, 60, 80, 120):

            ring_files = [np.load(filename + 'ring_{}_gaussians_{}.npy'.format(ring, ring_size)) for filename in data_folders]

            output_ring_file = np.concatenate(ring_files)

            np.save(output_folder + 'ring_{}_gaussians_{}.npy'.format(ring, ring_size), output_ring_file)

# Merge head direction (rotation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_bottom_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_bottom_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_top_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_top_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_centre/']

merge_training_data(data_folders, output_folder)

# Merge grid code (translation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_bl_tr/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_tr_bl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_br_tl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_tl_br/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_lm_rm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_rm_lm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_bm_tm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_tm_bm/']

merge_training_data(data_folders, output_folder)

# Merge grid and hd data for comprehensive training set (we hope)

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_vis_hd_grid/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_all/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_all/']

merge_training_data(data_folders, output_folder)