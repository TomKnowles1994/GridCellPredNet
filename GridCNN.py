import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def load_npy_data(data_path, sample_count, images_file = 'images.npy', hd_file = 'networkOutput_gaussianised.npy', grid_file = 'gaussian_grid_code_30.npy', offset = 0, shuffle=True):

    img = np.load(data_path + images_file)
    
    print(img.shape)
    
    if img.shape[0] > sample_count:

        img = np.load(data_path + images_file)[offset:sample_count+offset]

        print(img.shape)

    hd_data = np.load(data_path + hd_file)

    print(hd_data.shape)

    if hd_data.shape[0] > sample_count:

        hd_data = np.load(data_path + hd_file)[offset:sample_count+offset]

        print(hd_data.shape)

    if img.shape[0] != hd_data.shape[0]:

        min_sample_count = min(img.shape[0], hd_data.shape[0])

        img = img[offset:min_sample_count+offset]
        print(img.shape)

        hd_data = hd_data[offset:min_sample_count+offset]
        print(hd_data.shape)

    grid_data = np.load(data_path + grid_file)

    print(grid_data.shape)

    if grid_data.shape[0] > img.shape[0]:

        grid_data = grid_data[offset:img.shape[0]+offset]

    print(grid_data.shape)

    if shuffle:
        # shuffle sequence of data but maintain visual-hd-grid alignment
        img, hd_data, grid_data = shuffle_in_sync(img, hd_data, grid_data)

    return img, hd_data, grid_data

def shuffle_in_sync(visual_data, hd_data, grid_data):

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_hd, shuffled_grid = visual_data[shared_indices], hd_data[shared_indices], grid_data[shared_indices]

    return shuffled_visual, shuffled_hd, shuffled_grid

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/training_data_all/' # Point this to the training data folder

save_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/guifen_experiment_reconstruction/GridCNN/' # Point this to where the checkpoints are to be saved

images, head_direction, grid_code = load_npy_data(data_path, 2000, shuffle = False)

def build_model():

    input_layer = keras.Input(shape = (45, 80, 3))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    output_hd = keras.layers.Dense(180)(x)
    output_grid_1 = keras.layers.Dense(1000)(x)
    #output_grid_2 = keras.layers.Dense(1000)(x)
    #output_grid_3 = keras.layers.Dense(1000)(x)
    #output_grid_4 = keras.layers.Dense(1000)(x)
    #output_grid_5 = keras.layers.Dense(1000)(x)

    output_layers = [output_hd, output_grid_1]#, output_grid_2, output_grid_3, output_grid_4, output_grid_5]

    model = keras.Model(inputs = input_layer, outputs = output_layers, name = 'Grid_HD_Model')

    model.compile(loss='mse')

    model.compile(
    optimizer=keras.optimizers.Adam(1e-6),
    loss="mse"
    )

    return model

model = build_model()

val_data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/' # Point this to the validation set

val_images, val_head_direction, val_grid_code = load_npy_data(val_data_path, 2000, offset = 0, shuffle = False)

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

y_data = [head_direction, grid_code[:, 0:1000]]#, grid_code[:, 1000:2000], grid_code[:, 2000:3000], grid_code[:, 3000:4000], grid_code[:, 4000:5000]]

x_val = val_images

y_val = [val_head_direction, val_grid_code[:, 0:1000]]#, val_grid_code[:, 1000:2000], val_grid_code[:, 2000:3000], val_grid_code[:, 3000:4000], val_grid_code[:, 4000:5000]]

model.fit(x = images, y = y_data, validation_data = (x_val, y_val), batch_size = 10, epochs = 50, callbacks = [lr_schedule])

model.save_weights(save_path + 'main.ckpt')

model.load_weights(save_path + 'main.ckpt')

if True:
    dataset = val_data_path # Point to the dataset folder

    predictions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_real_data_6005_timestamped/conv/"

# Alternatively: "for dataset in (<comma-seperated dataset folders>)" for multiple folders:

    print("Creating Predictions for {} dataset".format(dataset))

    data_path = '{}'.format(dataset) # Point to dataset folder

    test_images, test_hd, test_grid_1 = load_npy_data(data_path, 3000, offset = 2000, shuffle = False)

    hd_predictions, grid_1_predictions = model.predict(x = test_images)

    np.save(predictions_folder + "hd.npy", hd_predictions)

    np.save(predictions_folder + "grid_1.npy", grid_1_predictions)

fig, ax = plt.subplots(2,1)

plt.subplots_adjust(hspace = 0.4)

ax[0].plot(test_hd[0, :], label = 'Ground Truth')
ax[0].plot(hd_predictions[0, :], label = 'Prediction')
ax[0].set_title("HD Predictions")
ax[0].legend()

ax[1].plot(test_grid_1[0, :], label = 'Ground Truth')
ax[1].plot(grid_1_predictions[0, :], label = 'Prediction')
ax[1].set_title("Grid 1 Predictions")
ax[1].legend()

plt.show()