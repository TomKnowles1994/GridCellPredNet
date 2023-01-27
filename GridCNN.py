import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def load_npy_data(data_path, sample_count, offset = 0, shuffle = False):

    img = np.load(data_path + 'images.npy')
    
    print(img.shape)
    
    if img.shape[0] > sample_count:

        img = img[offset:sample_count+offset]

        print(img.shape)

    hd_data = np.load(data_path + 'networkOutput_gaussianised.npy')

    print(hd_data.shape)

    if hd_data.shape[0] > sample_count:

        hd_data = hd_data[offset:sample_count+offset]

        print(hd_data.shape)

    if img.shape[0] != hd_data.shape[0]:

        min_sample_count = min(img.shape[0], hd_data.shape[0])

        img = img[offset:min_sample_count+offset]
        print(img.shape)

        hd_data = hd_data[offset:min_sample_count+offset]
        print(hd_data.shape)

    grid_data = np.concatenate([np.load(data_path + 'gaussian_grid_code_{}.npy'.format(neuron_count)) for neuron_count in (30, 40, 60)], axis = 1)

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

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_vis_hd_grid/' # Point this to the training data folder

save_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/guifen_experiment_reconstruction/GridCNN/' # Point this to where the checkpoints are to be saved

images, head_direction, grid_code = load_npy_data(data_path, 3000, shuffle = True)

def build_model():

    input_layer = keras.Input(shape = (45, 80, 3))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    output_hd = keras.layers.Dense(180, activation='relu', name = 'HD')(x)
    output_grid_1 = keras.layers.Dense(1000, activation='relu', name = 'G30')(x)
    output_grid_2 = keras.layers.Dense(1000, activation='relu', name = 'G40')(x)
    output_grid_3 = keras.layers.Dense(1000, activation='relu', name = 'G60')(x)
    #output_grid_4 = keras.layers.Dense(1000, activation='relu', name = 'G80')(x)
    #output_grid_5 = keras.layers.Dense(1000, activation='relu', name = 'G120')(x)

    output_layers = [output_hd, output_grid_1, output_grid_2, output_grid_3]#, output_grid_4, output_grid_5]

    model = keras.Model(inputs = input_layer, outputs = output_layers, name = 'Grid_HD_Model')

    model.compile(loss='mse')

    model.compile(
    optimizer=keras.optimizers.Adam(1e-6),
    loss="mse"
    )

    return model

model = build_model()

val_data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/' # Point this to the validation set

val_images, val_head_direction, val_grid_code = load_npy_data(val_data_path, 2000, offset = 0, shuffle = True)

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

y_data = [head_direction, grid_code[:, 0:1000], grid_code[:, 1000:2000], grid_code[:, 2000:3000]]#, grid_code[:, 3000:4000], grid_code[:, 4000:5000]]

x_val = val_images

y_val = [val_head_direction, val_grid_code[:, 0:1000], val_grid_code[:, 1000:2000], val_grid_code[:, 2000:3000]]#, val_grid_code[:, 3000:4000], val_grid_code[:, 4000:5000]]

#model.fit(x = images, y = y_data, validation_data = (x_val, y_val), batch_size = 10, epochs = 100, verbose = 2)#, callbacks = [lr_schedule])

#model.save_weights(save_path + 'main.ckpt')

model.load_weights(save_path + 'main.ckpt')

if True:
    dataset = val_data_path # Point to the dataset folder

    predictions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_real_data_6005_timestamped/conv/"

# Alternatively: "for dataset in (<comma-seperated dataset folders>)" for multiple folders:

    print("Creating Predictions for {} dataset".format(dataset))

    data_path = '{}'.format(dataset) # Point to dataset folder

    test_images, test_hd, test_grid = load_npy_data(data_path, 3000, offset = 2000, shuffle = False)

    #hd_predictions, grid_1_predictions, grid_2_predictions, grid_3_predictions, grid_4_predictions, grid_5_predictions = model.predict(x = test_images)
    predictions = model.predict(x = test_images)

    np.save(predictions_folder + "hd.npy", predictions[0])

    np.save(predictions_folder + "grid_1.npy", predictions[1])
    np.save(predictions_folder + "grid_2.npy", predictions[2])
    np.save(predictions_folder + "grid_3.npy", predictions[3])
    #np.save(predictions_folder + "grid_4.npy", grid_4_predictions)
    #np.save(predictions_folder + "grid_5.npy", grid_5_predictions)

number_of_rows = 2 + test_grid.shape[1] // 1000

fig, ax = plt.subplots(number_of_rows, 5, figsize = (20, number_of_rows * 4))

plt.subplots_adjust(hspace = 0.4)

for c, column in enumerate(ax[0]):

    column.imshow(test_images[c * 100, :].reshape(45, 80, 3))
    column.set_axis_off()
    #column.set_title("Input Image")

for c, column in enumerate(ax[1]):

    if c == 0:
        column.plot(test_hd[c * 100, :], label = 'Ground Truth')
        column.plot(predictions[0][c * 100, :], label = 'Prediction')
        column.set_ylabel("HD Predictions")
    else:
        column.plot(test_hd[c * 100, :])#, label = 'Ground Truth')
        column.plot(predictions[0][c * 100, :])#, label = 'Prediction')
    #column.set_title("HD Predictions")
    #column.legend()

for i, j in enumerate(range(2, number_of_rows)):
    for c, column in enumerate(ax[j]):

        column.plot(test_grid[c * 100, i*1000:(i+1)*1000])#, label = 'Ground Truth')
        column.plot(predictions[i+1][c * 100, :])#, label = 'Prediction')
        if c == 0:
            column.set_ylabel("Grid {} Predictions".format(i+1))
        #column.set_title("Grid {} Predictions".format(i))
        #column.legend()

fig.legend()

plt.show()