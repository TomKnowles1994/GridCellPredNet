import numpy as np

def merge_training_data(data_folders, output_folder, trim_start = 0, trim_end = 0):

    # Rotation datasets often have their first few samples distorted by the robot translating to the rotation point
    # Translation datasets have the same issue, but also may collide or merge with obstacles towards the end
    
    output_image_file = np.concatenate([np.load(filename + 'images.npy')[trim_start:-(trim_end+1)] for filename in data_folders])

    np.save(output_folder + 'images.npy', output_image_file)

    output_head_direction_file = np.concatenate([np.load(filename + 'networkOutput_gaussianised.npy')[trim_start:-(trim_end+1)] for filename in data_folders])

    np.save(output_folder + 'networkOutput_gaussianised.npy', output_head_direction_file)

    for ring in (1, 2, 3):
        for ring_size in (30, 40, 60, 80, 120):
            
            output_ring_file = np.concatenate([np.load(filename + 'ring_{}_gaussians_{}.npy'.format(ring, ring_size))[trim_start:-(trim_end+1)] for filename in data_folders])

            np.save(output_folder + 'ring_{}_gaussians_{}.npy'.format(ring, ring_size), output_ring_file)

# Original (Guifen-like) arena

# Merge head direction (rotation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_bottom_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_bottom_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_top_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_top_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_centre/']

#merge_training_data(data_folders, output_folder, trim_start = 15)

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

#merge_training_data(data_folders, output_folder, trim_start = 15, trim_end = 15)

# Merge grid and hd data for comprehensive training set (we hope)

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_vis_hd_grid/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_all/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_all/']

#merge_training_data(data_folders, output_folder)


# Dome arena

# Merge head direction (rotation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_bottom_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_bottom_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_top_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_top_right/']

#merge_training_data(data_folders, output_folder, trim_start = 15)

# Merge grid code (translation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_bl_tr/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_tr_bl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_br_tl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_tl_br/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_lm_rm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_rm_lm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_bm_tm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_tm_bm/']

#merge_training_data(data_folders, output_folder, trim_start = 15, trim_end = 15)

# Merge grid and hd data for comprehensive training set (we hope)

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_vis_hd_grid/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_all/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_all/']

#merge_training_data(data_folders, output_folder)


# Dome arena

# Merge head direction (rotation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_bottom_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_bottom_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_top_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_top_right/']

#merge_training_data(data_folders, output_folder, trim_start = 15)

# Merge grid code (translation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_all/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_bl_tr/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_tr_bl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_br_tl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_tl_br/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_lm_rm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_rm_lm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_bm_tm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_tm_bm/']

#merge_training_data(data_folders, output_folder, trim_start = 15, trim_end = 15)

# Merge grid and hd data for comprehensive training set (we hope)

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_vis_hd_grid/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_all/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_all/']

#merge_training_data(data_folders, output_folder)


# Oracle arena

# Merge head direction (rotation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_rotation/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/centre/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/bottom_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/bottom_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/top_left/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/top_right/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/centre_5x/']

merge_training_data(data_folders, output_folder, trim_start = 20)

# Merge grid code (translation) data

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_translation/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/bl_tr/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/tr_bl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/br_tl/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/tl_br/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/lm_rm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/rm_lm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/bm_tm/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/tm_bm/']

merge_training_data(data_folders, output_folder, trim_start = 20, trim_end = 40)

# Merge grid and hd data for comprehensive training set (we hope)

output_folder = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_vis_hd_grid/'

data_folders = ['C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_rotation/',
                'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_translation/']

merge_training_data(data_folders, output_folder)