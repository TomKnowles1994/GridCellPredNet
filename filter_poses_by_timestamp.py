import numpy as np

data_path = "C:\\Users\\Tom\\Downloads\\HBP\\multimodalplacerecognition_datasets\\whiskeye_guifen_datasets\\real_data_6005_timestamped\\"

timestamps = np.loadtxt(data_path + 'image_timestamps.csv', delimiter = ',', skiprows = 1) # Format should be (Time)

poses = np.loadtxt(data_path + 'body_poses.csv', delimiter = ',', skiprows = 1) # Format should be (Time, X, Y, Z)

print(poses.shape)

index = np.searchsorted(poses[:, 0], timestamps)

print(index)

filtered_poses = poses[index, :]

print(filtered_poses[:5])

np.save(data_path + 'filtered_body_poses.npy', filtered_poses)