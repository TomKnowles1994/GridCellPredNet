import numpy as np

data_root_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/"

data_paths_rotation = [     data_root_folder + "training_data_gazebo_3_3000/training_data_centre/",
                            data_root_folder + "training_data_gazebo_3_3000/training_data_top_left/",
                            data_root_folder + "training_data_gazebo_3_3000/training_data_top_right/",
                            data_root_folder + "training_data_gazebo_3_3000/training_data_bottom_left/",
                            data_root_folder + "training_data_gazebo_3_3000/training_data_bottom_right/",
                            data_root_folder + "training_data_gazebo_3_3000/training_data_centre_5x/"]

data_paths_translation = [  data_root_folder + "training_data_translation_sets_3000/training_data_bl_tr/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_tr_bl/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_br_tl/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_tl_br/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_lm_rm/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_rm_lm/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_bm_tm/",
                            data_root_folder + "training_data_translation_sets_3000/training_data_tm_bm/"]

data_paths = []
data_paths.extend(data_paths_translation)
data_paths.extend(data_paths_rotation)

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

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_all/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy')[15:] for filename in data_paths_rotation]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)

# Both

output_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_vis_hd_grid/"

filtered_body_pose_files = [np.load(filename + 'filtered_body_poses.npy') for filename in data_paths]

output_filtered_body_pose_file = np.concatenate(filtered_body_pose_files)

np.save(output_folder + 'filtered_body_poses.npy', output_filtered_body_pose_file)