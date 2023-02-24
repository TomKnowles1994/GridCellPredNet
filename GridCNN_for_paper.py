import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data_type = 'xy_theta'

# One of:-

# - 'xy_theta' : filtered pose data (X, Y) + (Theta)
# - 'rings' : ring coordinates and head direction (r1, r2, r3)
# - 'grid_index' : grid cell index
# - 'grid_gauss' : grid cell gaussian

training_data_type = 'rat'

# One of:-

# - 'cardinal' : contrived dataset built from robot rotating at several points plus moving along cardinal directions
# - 'rat' : dataset gathered from rat foraging (Guifen et al.)

ring_scale = 30

# Int from (30, 40, 60, 80, 120)

# Only to be used with data_type == 'rings'

grid_scales = (30,)#, 40, 60, 80, 120)

# Tuple of up to five from (30, 40, 60, 80, 120)

# Only used with data_type in ['grid_index', 'grid_gauss']

test_plot = False

training_epochs = 300

run_type = 'train'

# 'train' or 'test'

train_efficientnet = False

# True or False: add a further stage of training of the overall network, at a miniscule learning rate

data_root = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/'

# Plot incoming data, to check for data errors?

def load_npy_data(data_path, sample_count, offset = 0, shuffle = False):

    img = np.load(data_path + 'images.npy')[offset:sample_count+offset] * 255 # EfficientNetBX need images in 0-255 range

    print(img.shape)

    hd_data = np.load(data_path + 'networkOutput_gaussianised.npy')[offset:sample_count+offset]

    print(hd_data.shape)

    if img.shape[0] != hd_data.shape[0]:

        min_sample_count = min(img.shape[0], hd_data.shape[0])

        img = img[:min_sample_count]
        print(img.shape)

        hd_data = hd_data[:min_sample_count]
        print(hd_data.shape)

    if data_type == 'xy_theta':

        grid_data = np.load(data_path + 'filtered_body_poses.npy')[offset:sample_count+offset, 1:]

        print("Grid data max: {}".format(np.max(grid_data, axis = 0)))
        print("Grid data min: {}".format(np.min(grid_data, axis = 0)))

        grid_data[:, 2][grid_data[:, 2] < 0] = grid_data[:, 2][grid_data[:, 2] < 0] + 2 * np.pi

        print("Grid data max: {}".format(np.max(grid_data, axis = 0)))
        print("Grid data min: {}".format(np.min(grid_data, axis = 0)))

        grid_data = (grid_data - np.min(grid_data, axis = 0)) / (np.max(grid_data, axis = 0) - np.min(grid_data, axis = 0))

        print("Grid data max: {}".format(np.max(grid_data, axis = 0)))
        print("Grid data min: {}".format(np.min(grid_data, axis = 0)))

    elif data_type == 'rings':

        grid_data = np.concatenate([np.load(data_path + 'ring_{}_gaussians_{}.npy'.format(i, ring_scale))[offset:sample_count+offset] for i in (1,2,3)], axis = 1)
    
    elif data_type == 'grid_index':
    
        grid_data = np.concatenate([np.load(data_path + 'index_grid_code_{}.npy'.format(neuron_count))[offset:sample_count+offset] for neuron_count in grid_scales], axis = 1)
    
    elif data_type == 'grid_gauss':

        grid_data = np.concatenate([np.load(data_path + 'gaussian_grid_code_{}.npy'.format(neuron_count))[offset:sample_count+offset] for neuron_count in grid_scales], axis = 1)

    else:

        raise ValueError

    print(grid_data.shape)

    if test_plot:

        if data_type == 'xy_theta':

            fig, ax = plt.subplots(1,2)

            ax[0].plot(grid_data[:,0], grid_data[:,1])

            ax[1].plot(grid_data[:,2])

        elif data_type == 'rings':

            fig, axes = plt.subplots(1,3)

            for i, ax in enumerate(axes):

                ax.plot(grid_data[i*100,i*ring_scale:(i+1)*ring_scale])

        elif data_type in ['grid_index', 'grid_gauss']:

            fig, axes = plt.subplots(1,5)

            for i, ax in enumerate(axes):

                ax.plot(grid_data[i*100,i*1000:(i+1)*1000])

        plt.show()

    if shuffle:
        # shuffle sequence of data but maintain visual-hd-grid alignment
        img, hd_data, grid_data = shuffle_in_sync(img, hd_data, grid_data)

    return img, hd_data, grid_data

def shuffle_in_sync(visual_data, hd_data, grid_data):

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_hd, shuffled_grid = visual_data[shared_indices], hd_data[shared_indices], grid_data[shared_indices]

    return shuffled_visual, shuffled_hd, shuffled_grid

def extend_model(base_model, trainable, learning_rate = 1e-10):

    input_layer = keras.Input(shape = (224, 224, 3), name = 'Input')

    x = base_model(input_layer, training = trainable)

    x = keras.layers.Dense(features.shape[1], activation='relu', name = 'ExtraDense1')(x)
    x = keras.layers.Dense(features.shape[1]//4, activation='relu', name = 'ExtraDense2')(x)
    x = keras.layers.Dense(features.shape[1]//16, activation='relu', name = 'ExtraDense3')(x)

    if data_type == 'xy_theta':

        output_xy = keras.layers.Dense(2, activation='relu', name = 'XY')(x)
        output_theta = keras.layers.Dense(1, activation='relu', name = 'Theta')(x)
        # output_pose = keras.layers.Dense(3, activation='relu', name = 'Pose')(x)
        model = keras.Model(inputs = input_layer, outputs = [output_xy, output_theta])
        # model = keras.Model(inputs = input_layer, outputs = output_pose)

    elif data_type == 'rings':

        output_rings = []
        ring_labels = ('R1', 'R2', 'R3')
        for label in ring_labels:
            output_rings.append(keras.layers.Dense(ring_scale, activation='relu', name = label)(x))

        model = keras.Model(inputs = input_layer, outputs = output_rings)

    elif data_type in ['grid_index', 'grid_gauss']:

        output_hd = keras.layers.Dense(180, activation='relu', name = 'HD')(x)
        output_grid = []
        grid_layer_labels = ('G30', 'G40', 'G60', 'G80', 'G120')
        for scale, label in zip(grid_scales, grid_layer_labels):
            output_grid.append(keras.layers.Dense(1000, activation='relu', name = label)(x))

        model = keras.Model(inputs = input_layer, outputs = [output_hd, *output_grid])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate),
        loss = 'mse'
    )

    return model

def feature_input_model():

    input_layer = keras.Input(shape = (features.shape[1],), name = 'Features')

    print(features.shape[1])
    print(features.shape[1]//4)
    print(features.shape[1]//16)

    x = keras.layers.Dense(features.shape[1], activation='relu', name = 'ExtraDense1')(input_layer)
    x = keras.layers.Dense(features.shape[1]//4, activation='relu', name = 'ExtraDense2')(x)
    x = keras.layers.Dense(features.shape[1]//16, activation='relu', name = 'ExtraDense3')(x)
    x = keras.layers.Dense(features.shape[1]//64, activation='relu', name = 'ExtraDense4')(x)

    if data_type == 'xy_theta':

        output_xy = keras.layers.Dense(2, activation='relu', name = 'XY')(x)
        output_theta = keras.layers.Dense(1, activation='relu', name = 'Theta')(x)
        # output_pose = keras.layers.Dense(3, activation='relu', name = 'Pose')(x)
        model = keras.Model(inputs = input_layer, outputs = [output_xy, output_theta])
        # model = keras.Model(inputs = input_layer, outputs = output_pose)

    elif data_type == 'rings':

        output_rings = []
        ring_labels = ('R1', 'R2', 'R3')
        for label in ring_labels:
            output_rings.append(keras.layers.Dense(ring_scale, activation='relu', name = label)(x))

        model = keras.Model(inputs = input_layer, outputs = output_rings)

    elif data_type in ['grid_index', 'grid_gauss']:

        output_hd = keras.layers.Dense(180, activation='relu', name = 'HD')(x)
        output_grid = []
        grid_layer_labels = ('G30', 'G40', 'G60', 'G80', 'G120')
        for scale, label in zip(grid_scales, grid_layer_labels):
            output_grid.append(keras.layers.Dense(1000, activation='relu', name = label)(x))

        model = keras.Model(inputs = input_layer, outputs = [output_hd, *output_grid])

    model.compile(
        optimizer = keras.optimizers.Adam(1e-7),
        loss = 'mse'
    )

    return model

def load_inbuilt_model(trainable = False):

    model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling='max')

    model.trainable = trainable

    return model

def basic_CNN():

    input_layer = keras.Input(shape = (45, 80, 3), name = 'Input')

    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(3**5, activation='relu', name = 'ExtraDense2')(x)
    x = keras.layers.Dense(3**4, activation='relu', name = 'ExtraDense3')(x)
    x = keras.layers.Dense(3**3, activation='relu', name = 'ExtraDense4')(x)

    #print(x.shape[0]//4)
    #print(x.shape[0]//16)
    #print(x.shape[0]//64)

    if data_type == 'xy_theta':

        output_xy = keras.layers.Dense(2, activation='relu', name = 'XY')(x)
        output_theta = keras.layers.Dense(1, activation='relu', name = 'Theta')(x)
        # output_pose = keras.layers.Dense(3, activation='relu', name = 'Pose')(x)
        model = keras.Model(inputs = input_layer, outputs = [output_xy, output_theta])
        # model = keras.Model(inputs = input_layer, outputs = output_pose)

    elif data_type == 'rings':

        output_rings = []
        ring_labels = ('R1', 'R2', 'R3')
        for label in ring_labels:
            output_rings.append(keras.layers.Dense(ring_scale, activation='relu', name = label)(x))

        model = keras.Model(inputs = input_layer, outputs = output_rings)

    elif data_type in ['grid_index', 'grid_gauss']:

        output_hd = keras.layers.Dense(180, activation='relu', name = 'HD')(x)
        output_grid = []
        grid_layer_labels = ('G30', 'G40', 'G60', 'G80', 'G120')
        for scale, label in zip(grid_scales, grid_layer_labels):
            output_grid.append(keras.layers.Dense(1000, activation='relu', name = label)(x))

        model = keras.Model(inputs = input_layer, outputs = [output_hd, *output_grid])

    model.compile(
        optimizer = keras.optimizers.Adam(1e-7),
        loss = 'mse'
    )

    return model

if training_data_type == 'cardinal':

    data_path = data_root + 'training_data_vis_hd_grid/' # Point this to the training data folder
    images, head_direction, grid_code = load_npy_data(data_path, 3000, shuffle = False)
    
    val_data_path = data_root + 'training_data_vis_hd_grid/' # Point this to the validation set
    val_images, val_head_direction, val_grid_code = load_npy_data(val_data_path, 1000, offset = len(images), shuffle = False)

    save_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/oracle_at_delphi/GridCNN_{}/{}/'.format(training_data_type, data_type) # Point this to where the checkpoints are to be saved


elif training_data_type == 'rat':

    data_path = data_root + 'real_rat/' # Point this to the training data folder
    images, head_direction, grid_code = load_npy_data(data_path, 3000, shuffle = False)

    val_data_path = data_root + 'real_rat/' # Point this to the validation set
    val_images, val_head_direction, val_grid_code = load_npy_data(val_data_path, 1000, offset = len(images), shuffle = False)

    save_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/oracle_at_delphi/GridCNN_{}/{}/'.format(training_data_type, data_type) # Point this to where the checkpoints are to be saved

base_model = load_inbuilt_model()

features = base_model.predict(images)

np.save(save_path + 'EfficientNetB0_training_features.npy', features)

model = feature_input_model()

# fig, ax = plt.subplots(1,2)

# ax[0].plot(val_grid_code[:, 0], val_grid_code[:, 1])
# ax[1].plot(val_grid_code[:, 2])

# plt.show()

val_features = base_model.predict(val_images)

np.save(save_path + 'EfficientNetB0_validation_features.npy', val_features)

def scheduler(epoch, lr):
    if epoch < training_epochs // 2:
        return lr
    else:
        return lr * tf.math.exp(-0.01)
    
lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

if data_type == 'xy_theta':

    y_data = [grid_code[:, 0:2], grid_code[:, 2]]
    y_val = [val_grid_code[:, 0:2], val_grid_code[:, 2]]

elif data_type == 'rings':
    
    y_data = [grid_code[:,0:30], grid_code[:,30:60], grid_code[:,60:90]]
    y_val = [val_grid_code[:,0:30], val_grid_code[:,30:60], val_grid_code[:,60:90]]

elif data_type in ['grid_index', 'grid_gauss']:

    y_data = [head_direction]
    index_to_scale = {  30:  grid_code[:, 0:1000],
                        40:  grid_code[:, 1000:2000],
                        60:  grid_code[:, 2000:3000],
                        80:  grid_code[:, 3000:4000],
                        120: grid_code[:, 4000:5000]}
    for scale in grid_scales:
        if scale in index_to_scale.keys():
            y_data.append(index_to_scale[scale])

    y_val = [val_head_direction]
    index_to_scale = {  30:  val_grid_code[:, 0:1000],
                        40:  val_grid_code[:, 1000:2000],
                        60:  val_grid_code[:, 2000:3000],
                        80:  val_grid_code[:, 3000:4000],
                        120: val_grid_code[:, 4000:5000]}
    for scale in grid_scales:
        if scale in index_to_scale.keys():
            y_val.append(index_to_scale[scale])

x_val = val_features

early_stop = keras.callbacks.EarlyStopping(patience = 10)

if run_type == 'train':

    model.fit(x = features, y = y_data, validation_data = (x_val, y_val), batch_size = 10, epochs = training_epochs, verbose = 2, callbacks = [lr_schedule, early_stop])

    model.save_weights(save_path + 'main.ckpt')

    if train_efficientnet:

        base_model = load_inbuilt_model(trainable = True)

        model = feature_input_model()

        input_layer = keras.Input(shape = (224, 224, 3), name = 'Input')

        image_features = base_model(input_layer)

        output_xy, output_theta = model(image_features)

        print(output_xy)
        print(output_theta)

        model = keras.Model(inputs = input_layer, outputs = [output_xy, output_theta])

        model.compile(
            optimizer = keras.optimizers.Adam(1e-10),
            loss = 'mse'
        )

        model.fit(x = images, y = y_data, validation_data = (val_images, y_val), batch_size = 10, epochs = training_epochs, verbose = 2, callbacks = [lr_schedule, early_stop])

        model.save_weights(save_path + 'main.ckpt')

elif run_type == 'test':

    model.load_weights(save_path + 'main.ckpt')

else:

    raise ValueError

if True:

    dataset = val_data_path # Point to the dataset folder

    predictions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/oracle_at_delphi/"

    # Alternatively: "for dataset in (<comma-seperated dataset folders>)" for multiple folders:

    print("Creating Predictions for {} dataset".format(dataset))

    data_path = '{}'.format(dataset) # Point to dataset folder

    test_images, test_hd, test_grid = load_npy_data(data_path, 2000, offset = len(images) + len(val_images), shuffle = False)

    if test_plot:

        if data_type == 'xy_theta':

            fig, axes = plt.subplots(1,2)

            for ax in axes:

                ax.set_ylim(0, 1)

            axes[0].plot(test_grid[:,0], test_grid[:,1])

            axes[1].plot(test_grid[:,2])

        elif data_type == 'rings':

            fig, axes = plt.subplots(1,3)

            for i, ax in enumerate(axes):

                ax.plot(test_grid[i*100,i*ring_scale:(i+1)*ring_scale])


        elif data_type in ['grid_index', 'grid_gauss']:

            fig, axes = plt.subplots(1,5)

            for i, ax in enumerate(axes):

                ax.set_ylim(0, 1)

                ax.plot(test_grid[i*100,i*1000:(i+1)*1000])

        plt.show()

    test_features = base_model.predict(test_images)

    np.save(save_path + 'EfficientNetB0_test_features.npy', test_features)

    predictions = model.predict(x = test_features)

    if data_type == 'xy_theta':

        np.save(predictions_folder + "xy.npy", predictions[0])
        np.save(predictions_folder + "theta.npy", predictions[1])

    elif data_type == 'rings':

        np.save(predictions_folder + "r1.npy", predictions[0])
        np.save(predictions_folder + "r2.npy", predictions[1])
        np.save(predictions_folder + "r3.npy", predictions[2])

    elif data_type in ['grid_index', 'grid_gauss']:

        np.save(predictions_folder + "hd.npy", predictions[0])

        for i, prediction in enumerate(predictions):

            np.save(predictions_folder + "grid_{}.npy".format(i+1), prediction)

if data_type == 'xy_theta':

    number_of_rows = 2

elif data_type == 'rings':

    number_of_rows = 4

elif data_type in ['grid_index', 'grid_gauss']:

    number_of_rows = 2 + test_grid.shape[1] // 1000

fig, ax = plt.subplots(number_of_rows, 5, figsize = (20, number_of_rows * 4))

plt.subplots_adjust(hspace = 0.4)

for c, column in enumerate(ax[0]):

    column.imshow(test_images[c * 100, :].reshape(224, 224, 3) / 255)
    column.set_axis_off()
    #column.set_title("Input Image")

if data_type == 'xy_theta':

    for c, column in enumerate(ax[1]):

        if c == 0:
            X = test_grid[c * 100, 0]
            Y = test_grid[c * 100, 1]
            rescaled_theta = test_grid[c * 100, 2] * 2 * np.pi
            U = np.cos(rescaled_theta)
            V = np.sin(rescaled_theta)

            print(X)
            column.scatter(X, Y, color = 'black')
            column.quiver(X, Y, U, V, angles = 'xy', color = 'black', scale = 8, label = 'Ground Truth')
            column.text(X, Y, "Th:{}".format(np.around(rescaled_theta, 2)))

            X = predictions[0][c * 100, 0]
            Y = predictions[0][c * 100, 1]
            rescaled_theta = predictions[1][c * 100, 0] * 2 * np.pi
            U = np.cos(rescaled_theta)
            V = np.sin(rescaled_theta)

            print(X)
            column.scatter(X, Y, color = 'red')
            column.quiver(X, Y, U, V, angles = 'xy', color = 'red', scale = 8, label = 'Prediction')
            column.text(X, Y, "Th:{}".format(np.around(rescaled_theta, 2)))
            column.set_ylabel("XY Theta Predictions")
            column.set_xlim(-0.1, 1.1)
            column.set_ylim(-0.1, 1.1)
        else:
            X = test_grid[c * 100, 0]
            Y = test_grid[c * 100, 1]
            rescaled_theta = test_grid[c * 100, 2] * 2 * np.pi
            U = np.cos(rescaled_theta)
            V = np.sin(rescaled_theta)

            column.scatter(X, Y, color = 'black')
            column.quiver(X, Y, U, V, angles = 'xy', color = 'black', scale = 8)
            column.text(X, Y, "Th:{}".format(np.around(rescaled_theta, 2)))

            X = predictions[0][c * 100, 0]
            Y = predictions[0][c * 100, 1]
            rescaled_theta = predictions[1][c * 100, 0] * 2 * np.pi
            U = np.cos(rescaled_theta)
            V = np.sin(rescaled_theta)

            column.scatter(X, Y, color = 'red')
            column.quiver(X, Y, U, V, angles = 'xy', color = 'red', scale = 8)
            column.text(X, Y, "Th:{}".format(np.around(rescaled_theta, 2)))
            column.set_xlim(-0.1, 1.1)
            column.set_ylim(-0.1, 1.1)

elif data_type == 'rings':

    for i, j in enumerate(range(1, number_of_rows)):
        for c, column in enumerate(ax[j]):

            column.plot(test_grid[c * 100, i*30:(i+1)*30])
            column.plot(predictions[i][c * 100, :])
            if c == 0:
                column.set_ylabel("Ring {} Predictions".format(i+1))

elif data_type in ['grid_index', 'grid_gauss']:

    for c, column in enumerate(ax[1]):

        if c == 0:
            column.plot(test_hd[c * 100, :], label = 'Ground Truth')
            column.plot(predictions[0][c * 100, :], label = 'Prediction')
            column.set_ylabel("HD Predictions")
        else:
            column.plot(test_hd[c * 100, :])
            column.plot(predictions[0][c * 100, :])

    for i, j in enumerate(range(2, number_of_rows)):
        for c, column in enumerate(ax[j]):

            column.plot(test_grid[c * 100, i*1000:(i+1)*1000])
            column.plot(predictions[i+1][c * 100, :])
            if c == 0:
                column.set_ylabel("Grid {} Predictions".format(i+1))

fig.legend()

plt.show()

fig, ax = plt.subplots(2,1)

if data_type == 'xy_theta':

    hd_mse = np.mean(((predictions[1] - np.max(test_hd, axis = 0)) ** 2), axis = 1)

elif data_type in ['grid_index', 'grid_gauss']:

    hd_mse = np.mean((predictions[0] - test_hd) ** 2, axis = 1)

print(hd_mse)

if data_type == 'xy_theta':

    grid_mse = np.mean((predictions[0] - test_grid[:,0:2]) ** 2, axis = 1)

elif data_type == ['grid_index', 'grid_gauss']:

    grid_mse = np.mean([((prediction - test) ** 2) for prediction, test in zip(predictions[1:], test_grid[:,::1000])], axis = 1)

print(grid_mse)

total_mse = np.sum([hd_mse, grid_mse])
print(total_mse)

ax[0].plot(hd_mse)
ax[1].plot(grid_mse)

plt.show()