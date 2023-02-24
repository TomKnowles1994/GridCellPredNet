import numpy as np

# Original (Guifen-like) arena

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/"

data_paths_rotation = [     data_root_folder + "training_data_rotation_sets_3000/training_data_centre/",
                            data_root_folder + "training_data_rotation_sets_3000/training_data_top_left/",
                            data_root_folder + "training_data_rotation_sets_3000/training_data_top_right/",
                            data_root_folder + "training_data_rotation_sets_3000/training_data_bottom_left/",
                            data_root_folder + "training_data_rotation_sets_3000/training_data_bottom_right/",
                            data_root_folder + "training_data_rotation_sets_3000/training_data_centre_5x/"]

data_paths_translation = [  data_root_folder + "training_data_translation_sets_3000/training_data_bl_tr/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_tr_bl/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_br_tl/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_tl_br/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_lm_rm/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_rm_lm/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_bm_tm/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_tm_bm/"]

data_path_real = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

data_paths = []
data_paths.extend(data_paths_translation)
data_paths.extend(data_paths_rotation)
data_paths.append(data_path_real)

for data_path in data_paths:

    timestamps = np.loadtxt(data_path + 'image_timestamps.csv', delimiter = ',', skiprows = 1) # Format should be (Time)

    poses = np.loadtxt(data_path + 'body_poses.csv', delimiter = ',', skiprows = 1) # Format should be (Time, X, Y, Z)

    index = np.searchsorted(poses[:, 0], timestamps)

    filtered_poses = poses[index, :]

    np.save(data_path + 'filtered_body_poses.npy', filtered_poses)

### Merge together as needed

# Translation sets

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_all/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[15:-15] for filename in data_paths_translation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Rotation sets

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/training_data_all/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[15:] for filename in data_paths_rotation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Both

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_vis_hd_grid/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy') for filename in data_paths]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)


# Dome arena

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/"

# Missing a regular 'centre' dataset

data_paths_rotation = [     data_root_folder + "training_dome_rotation_sets_3000/training_dome_top_left/",
                            data_root_folder + "training_dome_rotation_sets_3000/training_dome_top_right/",
                            data_root_folder + "training_dome_rotation_sets_3000/training_dome_bottom_left/",
                            data_root_folder + "training_dome_rotation_sets_3000/training_dome_bottom_right/",
                            data_root_folder + "training_dome_rotation_sets_3000/training_dome_centre_5x/"]

data_paths_translation = [  data_root_folder + "training_dome_translation_sets_3000/training_dome_bl_tr/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_tr_bl/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_br_tl/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_tl_br/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_lm_rm/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_rm_lm/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_bm_tm/",
                            data_root_folder + "training_dome_translation_sets_3000/training_dome_tm_bm/"]

data_path_real = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_dome_6005_timestamped/"

data_paths = []
data_paths.extend(data_paths_translation)
data_paths.extend(data_paths_rotation)
data_paths.append(data_path_real)

for data_path in data_paths:

    timestamps = np.loadtxt(data_path + 'image_timestamps.csv', delimiter = ',', skiprows = 1) # Format should be (Time)

    poses = np.loadtxt(data_path + 'body_poses.csv', delimiter = ',', skiprows = 1) # Format should be (Time, X, Y, Z)

    index = np.searchsorted(poses[:, 0], timestamps)

    filtered_poses = poses[index, :]

    np.save(data_path + 'filtered_body_poses.npy', filtered_poses)

### Merge together as needed

# Translation sets

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/training_dome_all/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[15:-15] for filename in data_paths_translation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Rotation sets

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/training_dome_all/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[15:] for filename in data_paths_rotation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Both

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_vis_hd_grid/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy') for filename in data_paths]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)


# Oracle arena

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/"

data_paths_rotation =   [   data_root_folder + "training_data/centre/",
                            data_root_folder + "training_data/top_left/",
                            data_root_folder + "training_data/top_right/",
                            data_root_folder + "training_data/bottom_left/",
                            data_root_folder + "training_data/bottom_right/",
                            data_root_folder + "training_data/centre_5x/"]
                            
data_paths_translation = [  data_root_folder + "training_data/bl_tr/",
                            data_root_folder + "training_data/tr_bl/",
                            data_root_folder + "training_data/br_tl/",
                            data_root_folder + "training_data/tl_br/",
                            data_root_folder + "training_data/lm_rm/",
                            data_root_folder + "training_data/rm_lm/",
                            data_root_folder + "training_data/bm_tm/",
                            data_root_folder + "training_data/tm_bm/"]

data_path_real = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/real_rat/"

data_paths = []
data_paths.extend(data_paths_translation)
data_paths.extend(data_paths_rotation)
data_paths.append(data_path_real)

for data_path in data_paths:

    timestamps = np.loadtxt(data_path + 'image_timestamps.csv', delimiter = ',', skiprows = 1) # Format should be (Time)

    poses = np.loadtxt(data_path + 'body_poses.csv', delimiter = ',', skiprows = 1) # Format should be (Time, X, Y, Z)

    index = np.searchsorted(poses[:, 0], timestamps)

    filtered_poses = poses[index, :]

    np.save(data_path + 'filtered_body_poses.npy', filtered_poses)

### Merge together as needed

# Translation sets

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[20:-40] for filename in data_paths_translation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Rotation sets

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[20:] for filename in data_paths_rotation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Both

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_vis_hd_grid/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy') for filename in data_paths]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)