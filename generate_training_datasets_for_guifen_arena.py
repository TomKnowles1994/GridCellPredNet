import numpy as np
import pandas as pd

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/"

bottom_corner = (-3.0, -3.0)
top_corner = (3.0, 3.0)

x_coordinates = np.linspace(bottom_corner[0], top_corner[0], 1000)
y_coordinates = np.linspace(bottom_corner[1], top_corner[1], 1000)

theta = np.repeat(np.radians(-45), 1000)

translation_dataset = pd.DataFrame(data = {"X": x_coordinates, "Y": y_coordinates, "Theta": theta})
translation_dataset.to_csv(data_folder + "translation_dataset.csv", index = False)

centre =        (0.0, 0.0)
top_left =      (-2.0,  2.0)
top_right =     ( 2.0,  2.0)
bottom_left =   (-2.0, -2.0)
bottom_right =  (-2.0,  2.0)

theta = np.linspace(0, 2*np.pi, 1000)

x_coordinates = np.repeat(centre[0], 1000)
y_coordinates = np.repeat(centre[1], 1000)

rotation_dataset_centre =       pd.DataFrame(data = {"X": x_coordinates, "Y": y_coordinates, "Theta": theta})
rotation_dataset_centre.to_csv(data_folder + "rotation_dataset_centre.csv", index = False)

x_coordinates = np.repeat(top_left[0], 1000)
y_coordinates = np.repeat(top_left[1], 1000)

rotation_dataset_top_left =     pd.DataFrame(data = {"X": x_coordinates, "Y": y_coordinates, "Theta": theta})
rotation_dataset_top_left.to_csv(data_folder + "rotation_dataset_top_left.csv", index = False)

x_coordinates = np.repeat(top_right[0], 1000)
y_coordinates = np.repeat(top_right[1], 1000)

rotation_dataset_top_right =    pd.DataFrame(data = {"X": x_coordinates, "Y": y_coordinates, "Theta": theta})
rotation_dataset_top_right.to_csv(data_folder + "rotation_dataset_top_right.csv", index = False)

x_coordinates = np.repeat(bottom_left[0], 1000)
y_coordinates = np.repeat(bottom_left[1], 1000)

rotation_dataset_bottom_left =  pd.DataFrame(data = {"X": x_coordinates, "Y": y_coordinates, "Theta": theta})
rotation_dataset_bottom_left.to_csv(data_folder + "rotation_dataset_bottom_left.csv", index = False)

x_coordinates = np.repeat(bottom_right[0], 1000)
y_coordinates = np.repeat(bottom_right[1], 1000)

rotation_dataset_bottom_right = pd.DataFrame(data = {"X": x_coordinates, "Y": y_coordinates, "Theta": theta})
rotation_dataset_bottom_right.to_csv(data_folder + "rotation_dataset_bottom_right.csv", index = False)