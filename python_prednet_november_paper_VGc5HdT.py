"""
To train a Multi-modal Predictive coding Network (MuPNet) using visual head_direction data gathered from Physical and simulated WhiskEye robot
"""

import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Seed value
seed_value = 29384767

# 1. Set PYTHONHASHSEED environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set Python built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set numpy pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the Tensorflow pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

### Data Processing Functions

def load_npy_data(data_path, sample_start, sample_end, minibatch_size, shuffle=True):

    sample_count = sample_end - sample_start

    images_file = '/images.npy'

    #grid_file = '/body_pose_grid_cell_activity.npy'
    grid_file = '/PredNet_grid_data.npy'

    head_direction_file = '/networkOutput_gaussianised.npy'

    #tactile_file = '/whisker_data.npy'
    tactile_file = '/whisker_data.csv'

    visual_data = np.load(data_path + images_file)[sample_start:sample_end]#/255

    plt.imshow(visual_data[0])

    visual_data = visual_data.reshape(visual_data.shape[0], 10800) # flatten

    head_direction_data = np.load(data_path + head_direction_file)[sample_start:sample_end]#np.load(data_path + '/networkOutput.npy').T

    grid_data = np.load(data_path + grid_file)[sample_start:sample_end] # Grid data is at 50Hz, so must be subsampled

    # fig, ax = plt.subplots(3,1)

    # pre_reshape_grid = np.concatenate([grid_data[1000,0,:], grid_data[1000,1,:], grid_data[1000,2,:], grid_data[1000,3,:], grid_data[1000,4,:]], axis = 0)

    # ax[0].scatter(np.arange(5000), pre_reshape_grid, s = 4)

    # ax[1].scatter(np.arange(1000), grid_data[1000,0,:], s = 4)
    # ax[1].scatter(np.arange(1000), grid_data[1000,1,:], s = 4)
    # ax[1].scatter(np.arange(1000), grid_data[1000,2,:], s = 4)
    # ax[1].scatter(np.arange(1000), grid_data[1000,3,:], s = 4)
    # ax[1].scatter(np.arange(1000), grid_data[1000,4,:], s = 4)

    #grid_data = grid_data.reshape(grid_data.shape[0], 5000)

    # ax[2].scatter(np.arange(5000), grid_data[1000,:], s = 4)

    # plt.show()

    #grid_data = np.zeros(shape = (len(visual_data), 5000))

    #tactile_data = np.load(data_path + tactile_file)[sample_start:sample_end]
    tactile_data = np.loadtxt(data_path + tactile_file, delimiter= ',', skiprows = 1)[sample_start:sample_end]

    # If <dataset> doesn't divide evenly into minibatches, round it off

    if visual_data.shape[0] % minibatch_size is not 0:

        new_shape = visual_data.shape[0]//minibatch_size
        new_shape = new_shape * minibatch_size

        visual_data = visual_data[:new_shape]

    if head_direction_data.shape[0] % minibatch_size is not 0:

        new_shape = head_direction_data.shape[0]//minibatch_size
        new_shape = new_shape * minibatch_size

        head_direction_data = head_direction_data[:new_shape]

    if grid_data.shape[0] % minibatch_size is not 0:

        new_shape = grid_data.shape[0]//minibatch_size
        new_shape = new_shape * minibatch_size

        grid_data = grid_data[:new_shape]

    if tactile_data.shape[0] % minibatch_size is not 0:

        new_shape = tactile_data.shape[0]//minibatch_size
        new_shape = new_shape * minibatch_size

        tactile_data = tactile_data[:new_shape]

    if shuffle:
        # shuffle sequence of data but maintain visual-head_direction alignment
        visual_data, grid_data, head_direction_data, tactile_data = shuffle_in_sync(visual_data, grid_data, head_direction_data, tactile_data)

    print("Final image data shape: {}".format(visual_data.shape))
    print("Final HD data shape: {}".format(head_direction_data.shape))
    print("Final grid field data shape {}".format(grid_data.shape))
    print("Final tactile data shape {}".format(tactile_data.shape))

    #n_sample = visual_data.shape[0]

    return visual_data, grid_data, head_direction_data, tactile_data                          

def shuffle_in_sync(visual_data, grid_data, head_direction_data, tactile_data):
    #assert visual_data.shape[0] == head_direction_data.shape[0]

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_grid, shuffled_head_direction, shuffled_tactile = visual_data[shared_indices], grid_data[shared_indices], head_direction_data[shared_indices], tactile_data[shared_indices]

    return shuffled_visual, shuffled_grid, shuffled_head_direction, shuffled_tactile

### User-defined Parameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

sample_start = 0

sample_end = 3000

n_sample = sample_end - sample_start        # If you have collected your own dataset, you will need to determine how many samples where collected in the run
                                            # Alternatively, if you are using a built-in dataset, copy the sample number as described in the datasets' README

minibatch_sz = 10                           # Minibatch size. Can be left as default for physical data, for simulated data good numbers to try are 40, 50 and 100
                                            # Datasize size must be fully divisible by minibatch size

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped'               # Path to training data. Training data should be in .npy format:

save_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/guifen_experiment_reconstruction/real_data_Gc5'#trial1'               # Path to save trained model to (once trained)
load_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/guifen_experiment_reconstruction/real_data_Gc5'#trial1'               # Path to load trained model from (after training, or if already trained beforehand)

cause_save_path = save_path + '/causes'#trial1/causes'  # Path to save causes to (optional, for training diagnostics)
reconstruction_save_path = save_path + '/reconstructions'#trial1/reconstructions'  # Path to save reconstructons to (optional, for training diagnostics)

save_vis_causes = False
save_hd_causes = False
save_vis_hd_causes = False
save_grid_causes = False
save_vis_grid_hd_tac_causes = False

save_vis_reconstructions = False
save_vis_hd_reconstructions = False
save_grid_reconstructions = False
save_vis_grid_hd_tac_reconstructions = False

n_epoch = 200                               # Number of training epochs to generate model. Default is 200
                                            
shuffle_data = False                        # Do you want to shuffle the training data? Default is False

# Load the data from .mat files

#visual_data, head_direction_data = load_mat_data(data_path, shuffle_data)

# Alternatively, load the data from .npy files

visual_data, grid_data, head_direction_data, tactile_data = load_npy_data(data_path, sample_start, sample_end, minibatch_sz, shuffle_data)

### Model Hyperparameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

load_model = False                          # If True, load a previously trained model from load_path. If False, train from scratch.

inp_shape =    {"vis": visual_data.shape[1],
                "grid": grid_data.shape[2],
                "hd": head_direction_data.shape[1],
                "tac": tactile_data.shape[1]}

layers =   {"vis": [1000],
            "grid": [1000],
            "vis_grid_hd_tac": [300]}

cause_init =   {"vis": [0.1, 0.1],
                "grid": [0.1, 0.1],
                "vis_grid_hd_tac": [0.1]}

reg_causes =   {"vis": [0.0, 0.0],
                "grid": [0.0, 0.0],
                "vis_grid_hd_tac": [0.0]}

lr_causes =    {"vis": [0.0004, 0.0004],
                "grid": [0.0004, 0.0004],
                "hd": [0.0004, 0.0004],
                "tac": [0.0004, 0.0004],
                "vis_grid_hd_tac": [0.0004]}

reg_filters =  {"vis": [0.0, 0.0],
                "grid": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "hd": [0.0, 0.0],
                "tac": [0.0, 0.0],
                "vis_grid_hd_tac": [0.0, 0.0, 0.0, 0.0]}

lr_filters =   {"vis": [0.0001, 0.0001],
                "grid": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
                "hd": [0.0001, 0.0001],
                "tac": [0.0001, 0.0001],
                "vis_grid_hd_tac": [0.0001, 0.0001, 0.0001, 0.0001]}

class Network:
    def __init__(self, n_sample, minibatch_sz, inp_shape, layers, cause_init, reg_causes, lr_causes, reg_filters, lr_filters):

        # create placeholders
        self.x_vis = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['vis']])
        self.x_grid_0 = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['grid']])
        self.x_grid_1 = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['grid']])
        self.x_grid_2 = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['grid']])
        self.x_grid_3 = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['grid']])
        self.x_grid_4 = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['grid']])
        self.x_hd = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['hd']])
        self.x_tac = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['tac']])
        self.batch = tf.placeholder(tf.int32, shape=[])

        # create filters and causes for visual data
        self.vis_filters = []
        self.vis_causes = []
        for i in range(len(layers['vis'])):
            filter_name = 'vis_filter_%d' % i
            cause_name = 'vis_cause_%d' % i

            if i == 0:
                self.vis_filters += [tf.get_variable(filter_name, shape=[layers['vis'][i], inp_shape['vis']])]
            else:
                self.vis_filters += [tf.get_variable(filter_name, shape=[layers['vis'][i], layers['vis'][i-1]])]

            init = tf.constant_initializer(cause_init['vis'][i])
            self.vis_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['vis'][i]], initializer=init)]

        # create filters and causes for grid cells
        self.grid_filters = []
        self.grid_causes = []
        for i in range(len(layers['grid'])):
            cause_name = 'grid_cause_%d' % i

            if i == 0:
                filter_name = 'grid_filter_0_0'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_1'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_2'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_3'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_4'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
            else:
                filter_name = 'grid_filter_%d' % i
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], layers['grid'][i-1]])]

            init = tf.constant_initializer(cause_init['grid'][i])
            self.grid_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['grid'][i]], initializer=init)]

        # create filters and cause for latent space visual + head_direction + grid + tactile
        self.vis_grid_hd_tac_filters = []
        self.vis_grid_hd_tac_causes = []
        for i in range(len(layers['vis_grid_hd_tac'])):
            if i == 0:
                # add filters for vis
                filter_name = 'vis_grid_hd_tac->vis_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   layers['vis'][-1]])]

                # add filters for grid
                filter_name = 'vis_grid_hd_tac->grid_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   layers['grid'][-1]])]

                # add filters for hd
                filter_name = 'vis_grid_hd_tac->hd_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   inp_shape['hd']])]

                # add filters for grid
                filter_name = 'vis_grid_hd_tac->tac_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   inp_shape['tac']])]
                
            else:
                filter_name = 'vis_grid_hd_tac_filter_%d' % i
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   layers['vis_grid_hd_tac'][i - 1]])]

            cause_name = 'vis_grid_hd_tac_causes_%d' % i
            init = tf.constant_initializer(cause_init['vis_grid_hd_tac'][i])
            self.vis_grid_hd_tac_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['vis_grid_hd_tac'][i]], initializer=init)]

        # compute predictions
        current_batch = tf.range(self.batch * minibatch_sz, (self.batch + 1) * minibatch_sz)
        #print("Current Batch: {}".format(current_batch))
        # vis predictions
        self.vis_minibatch = []
        self.vis_predictions = []
        for i in range(len(layers['vis'])):
            self.vis_minibatch += [tf.gather(self.vis_causes[i], indices=current_batch, axis=0)]
            self.vis_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_minibatch[i], self.vis_filters[i]))]
            #print("Vis Predictions: {}".format(self.vis_predictions[i].get_shape()))

        # grid predictions
        self.grid_minibatch = []
        self.grid_predictions = []
        for i in range(len(layers['grid'])):
            self.grid_minibatch += [tf.gather(self.grid_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i+1]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i+2]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i+3]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i+4]))]
            else:
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i+4]))]

        # vis_grid_hd_tac predictions
        self.vis_grid_hd_tac_minibatch = []
        self.vis_grid_hd_tac_predictions = []
        for i in range(len(layers['vis_grid_hd_tac'])):
            self.vis_grid_hd_tac_minibatch += [tf.gather(self.vis_grid_hd_tac_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_minibatch[i], self.vis_grid_hd_tac_filters[i]))]  # vis prediction
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_minibatch[i], self.vis_grid_hd_tac_filters[i+1]))]  # grid prediction
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_minibatch[i], self.vis_grid_hd_tac_filters[i+2]))]  # hd prediction
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_minibatch[i], self.vis_grid_hd_tac_filters[i+3]))]  # tac prediction
            else:
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_minibatch[i], self.vis_grid_hd_tac_filters[i+3]))]

        # add ops for computing gradients for vis causes and for updating weights
        self.vis_bu_error = []
        self.vis_update_filter = []
        self.vis_cause_grad = []
        for i in range(len(layers['vis'])):
            if i == 0:
                self.vis_bu_error += [tf.losses.mean_squared_error(self.x_vis, self.vis_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.vis_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_minibatch[i - 1]), self.vis_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(layers['vis']) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_predictions[i+1]), self.vis_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_grid_hd_tac_predictions[0]), self.vis_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['vis'][i] * (self.vis_minibatch[i] ** 2)
            error_list = [self.vis_bu_error[i], td_error, reg_error]
            assert len(error_list) == 3
            self.vis_cause_grad += [tf.gradients([self.vis_bu_error[i], td_error, reg_error],
                                                          self.vis_minibatch[i])[0]]

            # ops for updating weights
            reg_error = reg_filters['vis'][i] * (self.vis_filters[i] ** 2)
            vis_filter_grad = tf.gradients([self.vis_bu_error[i], reg_error], self.vis_filters[i])[0]
            self.vis_update_filter += [
                tf.assign_sub(self.vis_filters[i], lr_filters['vis'][i] * vis_filter_grad)]

        # add ops for computing gradients for grid causes and for updating weights
        self.grid_bu_error = []
        self.grid_update_filter = []
        self.grid_cause_grad = []
        for i in range(len(layers['grid'])):
            if i == 0:
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_0, self.grid_predictions[0],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_1, self.grid_predictions[1],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_2, self.grid_predictions[2],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_3, self.grid_predictions[3],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_4, self.grid_predictions[4],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.grid_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_minibatch[i - 1]), self.grid_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(layers['grid']) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_predictions[5]), self.grid_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_grid_hd_tac_predictions[1]), self.grid_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['grid'][i] * (self.grid_minibatch[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_minibatch[i])
            if i == 0:
                error_list = [  self.grid_bu_error[i], self.grid_bu_error[i+1], self.grid_bu_error[i+2],
                                self.grid_bu_error[i+3], self.grid_bu_error[i+4], td_error, reg_error]
                assert len(error_list) == len(self.grid_bu_error) + 2
            else:
                error_list = [  self.grid_bu_error[i+4], td_error, reg_error]
                assert len(error_list) == 3
            self.grid_cause_grad += [tf.gradients(error_list, self.grid_minibatch[i])[0]]

            # ops for updating weights

            if i == 0:

                reg_error = reg_filters['grid'][i] * (self.grid_filters[i] ** 2)
                grid_filter_grad = tf.gradients([self.grid_bu_error[i], reg_error], self.grid_filters[i])[0]
                self.grid_update_filter += [
                    tf.assign_sub(self.grid_filters[i], lr_filters['grid'][i] * grid_filter_grad)]
                reg_error = reg_filters['grid'][i+1] * (self.grid_filters[i+1] ** 2)
                grid_filter_grad = tf.gradients([self.grid_bu_error[i+1], reg_error], self.grid_filters[i+1])[0]
                self.grid_update_filter += [
                    tf.assign_sub(self.grid_filters[i+1], lr_filters['grid'][i+1] * grid_filter_grad)]
                reg_error = reg_filters['grid'][i+2] * (self.grid_filters[i+2] ** 2)
                grid_filter_grad = tf.gradients([self.grid_bu_error[i+2], reg_error], self.grid_filters[i+2])[0]
                self.grid_update_filter += [
                    tf.assign_sub(self.grid_filters[i+2], lr_filters['grid'][i+2] * grid_filter_grad)]
                reg_error = reg_filters['grid'][i+3] * (self.grid_filters[i+3] ** 2)
                grid_filter_grad = tf.gradients([self.grid_bu_error[i+3], reg_error], self.grid_filters[i+3])[0]
                self.grid_update_filter += [
                    tf.assign_sub(self.grid_filters[i+3], lr_filters['grid'][i+3] * grid_filter_grad)]
                reg_error = reg_filters['grid'][i+4] * (self.grid_filters[i+4] ** 2)
                grid_filter_grad = tf.gradients([self.grid_bu_error[i+4], reg_error], self.grid_filters[i+4])[0]
                self.grid_update_filter += [
                    tf.assign_sub(self.grid_filters[i+4], lr_filters['grid'][i+4] * grid_filter_grad)]

            else:

                reg_error = reg_filters['grid'][i+4] * (self.grid_filters[i+4] ** 2)
                grid_filter_grad = tf.gradients([self.grid_bu_error[i+4], reg_error], self.grid_filters[i+4])[0]
                self.grid_update_filter += [
                    tf.assign_sub(self.grid_filters[i+4], lr_filters['grid'][i+4] * grid_filter_grad)]

        # add ops for computing gradients for vis_grid_hd_tac causes
        self.vis_grid_hd_tac_bu_error = []
        self.vis_grid_hd_tac_reg_error = []
        self.vis_grid_hd_tac_update_filter = []
        self.vis_grid_hd_tac_cause_grad = []
        for i in range(len(layers['vis_grid_hd_tac'])):
            if i == 0:
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_minibatch[-1]), self.vis_grid_hd_tac_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_minibatch[-1]), self.vis_grid_hd_tac_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_hd), self.vis_grid_hd_tac_predictions[i+2],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_tac), self.vis_grid_hd_tac_predictions[i+3],
                    reduction=tf.losses.Reduction.NONE)]

                self.vis_grid_hd_tac_reg_error += [reg_causes['vis_grid_hd_tac'][i] * (self.vis_grid_hd_tac_minibatch[i] ** 2)]
                # self.vis_hd_reg_error += [tf.keras.regularizers.l2(reg_causes['vis_hd'][i])(self.vis_hd_minibatch[i])]
                if len(layers['vis_grid_hd_tac']) > 1:
                    raise NotImplementedError
                else:
                    error_list = [  self.vis_grid_hd_tac_bu_error[i], self.vis_grid_hd_tac_bu_error[i+1], self.vis_grid_hd_tac_bu_error[i+2], 
                                    self.vis_grid_hd_tac_bu_error[i+3], self.vis_grid_hd_tac_reg_error[i]]
                    assert len(error_list) == len(self.vis_grid_hd_tac_bu_error) + 1 # No td_error here, since it is usually the topmost layer
                    self.vis_grid_hd_tac_cause_grad += [
                        tf.gradients(error_list, self.vis_grid_hd_tac_minibatch[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['vis_grid_hd_tac'][i] * (self.vis_grid_hd_tac_filters[i] ** 2)
                vis_grid_hd_tac_filter_grad = tf.gradients([self.vis_grid_hd_tac_bu_error[i], reg_error], self.vis_grid_hd_tac_filters[i])[0]
                self.vis_grid_hd_tac_update_filter += [
                    tf.assign_sub(self.vis_grid_hd_tac_filters[i], lr_filters['vis_grid_hd_tac'][i] * vis_grid_hd_tac_filter_grad, use_locking = True)]

                reg_error = reg_filters['vis_grid_hd_tac'][i+1] * (self.vis_grid_hd_tac_filters[i+1] ** 2)
                vis_grid_hd_tac_filter_grad = tf.gradients([self.vis_grid_hd_tac_bu_error[i+1], reg_error], self.vis_grid_hd_tac_filters[i+1])[0]
                self.vis_grid_hd_tac_update_filter += [
                    tf.assign_sub(self.vis_grid_hd_tac_filters[i+1],  lr_filters['vis_grid_hd_tac'][i+1] * vis_grid_hd_tac_filter_grad, use_locking = True)]

                reg_error = reg_filters['vis_grid_hd_tac'][i+2] * (self.vis_grid_hd_tac_filters[i+2] ** 2)
                vis_grid_hd_tac_filter_grad = tf.gradients([self.vis_grid_hd_tac_bu_error[i+2], reg_error], self.vis_grid_hd_tac_filters[i+2])[0]
                self.vis_grid_hd_tac_update_filter += [
                    tf.assign_sub(self.vis_grid_hd_tac_filters[i+2],  lr_filters['vis_grid_hd_tac'][i+2] * vis_grid_hd_tac_filter_grad, use_locking = True)]

                reg_error = reg_filters['vis_grid_hd_tac'][i+3] * (self.vis_grid_hd_tac_filters[i+3] ** 2)
                vis_grid_hd_tac_filter_grad = tf.gradients([self.vis_grid_hd_tac_bu_error[i+3], reg_error], self.vis_grid_hd_tac_filters[i+3])[0]
                self.vis_grid_hd_tac_update_filter += [
                    tf.assign_sub(self.vis_grid_hd_tac_filters[i+3],  lr_filters['vis_grid_hd_tac'][i+3] * vis_grid_hd_tac_filter_grad, use_locking = True)]
            else:
                raise NotImplementedError

        # add ops for updating causes
        self.vis_update_cause = []
        self.grid_update_cause = []
        self.vis_grid_hd_tac_update_cause = []
        with tf.control_dependencies(self.vis_cause_grad + self.grid_cause_grad + self.vis_grid_hd_tac_cause_grad):
            # vis modality
            for i in range(len(layers['vis'])):
                #print("Vis Causes: {}".format(self.vis_causes[i]))
                self.vis_update_cause += [tf.scatter_sub(self.vis_causes[i], indices=current_batch,
                                                                  updates=(lr_causes['vis'][i] * self.vis_cause_grad[i]), use_locking = True)]

            # grid modality
            for i in range(len(layers['grid'])):
                self.grid_update_cause += [tf.scatter_sub(self.grid_causes[i], indices=current_batch,
                                                                  updates=(lr_causes['grid'][i] * self.grid_cause_grad[i]), use_locking = True)]

            # vis_grid_hd_tac modality
            for i in range(len(layers['vis_grid_hd_tac'])):
                self.vis_grid_hd_tac_update_cause += [tf.scatter_sub(self.vis_grid_hd_tac_causes[i], indices=current_batch,
                                                                  updates=(lr_causes['vis_grid_hd_tac'][i] * self.vis_grid_hd_tac_cause_grad[i]), use_locking = True)]


def train():
    tf.compat.v1.reset_default_graph()

    completed_epoch = 0

    net = Network(n_sample, minibatch_sz, inp_shape, layers, cause_init, reg_causes, lr_causes, reg_filters, lr_filters)

    saver = tf.train.Saver()
    cause_epoch = 50
    config = tf.ConfigProto(device_count={'GPU': 1})
    with tf.Session(config=config) as sess:
        if load_model is True:
            saver.restore(sess, '%s/main.ckpt' % load_path)
        else:
            sess.run(tf.global_variables_initializer())

        if load_model is True:
            vis_epoch_loss = np.load('%s/vis_epoch_loss.npy' % load_path)
            assert completed_epoch == vis_epoch_loss.shape[0], 'Value of completed_epoch is incorrect'

            vis_epoch_loss = np.vstack([vis_epoch_loss, np.zeros((n_epoch, len(layers['vis'])))])

            vis_avg_activity = np.vstack(
                [np.load('%s/vis_avg_activity.npy' % load_path), np.zeros((n_epoch, len(layers['vis'])))])

            grid_epoch_loss = np.vstack(
                [np.load('%s/grid_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['grid']) + 1))])

            grid_avg_activity = np.vstack(
                [np.load('%s/grid_avg_activity.npy' % load_path), np.zeros((n_epoch, len(layers['grid'])))])

            vis_grid_hd_tac_epoch_loss = np.vstack(
                [np.load('%s/vis_grid_hd_tac_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['vis_grid_hd_tac']) + 1))])

            vis_grid_hd_tac_avg_activity = np.vstack(
                [np.load('%s/vis_grid_hd_tac_activity.npy' % load_path), np.zeros((n_epoch, len(layers['vis_grid_hd_tac'])))])

        else:
            vis_epoch_loss = np.zeros((n_epoch, len(layers['vis'])))
            grid_epoch_loss = np.zeros((n_epoch, len(layers['grid']) + 4))
            vis_grid_hd_tac_epoch_loss = np.zeros((n_epoch, len(layers['vis_grid_hd_tac']) + 3))

            vis_avg_activity = np.zeros((n_epoch, len(layers['vis'])))
            grid_avg_activity = np.zeros((n_epoch, len(layers['grid'])))
            vis_grid_hd_tac_avg_activity = np.zeros((n_epoch, len(layers['vis_grid_hd_tac'])))

        for i in range(n_epoch):
            current_epoch = completed_epoch + i

            n_batch = visual_data.shape[0] // minibatch_sz
            for j in range(n_batch):
                #print(j)
                visual_batch = visual_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]
                head_direction_batch = head_direction_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]
                grid_batch = grid_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :, :]
                tactile_batch = tactile_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]
                # print(visual_batch.shape)
                # print(head_direction_batch.shape)
                # print(grid_batch.shape)
                # print(tactile_batch.shape)

                # update causes
                for k in range(cause_epoch):
                    vis_cause, grid_cause, vis_grid_hd_tac_cause, vis_grad, grid_grad, vis_grid_hd_tac_grad, vis_grid_hd_tac_reg_error = sess.run(
                        [net.vis_update_cause, net.grid_update_cause, net.vis_grid_hd_tac_update_cause,
                        net.vis_cause_grad, net.grid_cause_grad, net.vis_grid_hd_tac_cause_grad, 
                        net.vis_grid_hd_tac_reg_error],
                        feed_dict={ net.x_vis: visual_batch, 
                                    net.x_grid_0: grid_batch[:, 0, :], 
                                    net.x_grid_1: grid_batch[:, 1, :],
                                    net.x_grid_2: grid_batch[:, 2, :],
                                    net.x_grid_3: grid_batch[:, 3, :],
                                    net.x_grid_4: grid_batch[:, 4, :],
                                    net.x_hd: head_direction_batch, 
                                    net.x_tac: tactile_batch, 
                                    net.batch: j})

                # for k in range(cause_epoch):
                #     vis_grad, grid_grad, vis_grid_hd_tac_grad, vis_grid_hd_tac_reg_error = sess.run(
                #         [net.vis_cause_grad, net.grid_cause_grad, net.vis_grid_hd_tac_cause_grad, 
                #         net.vis_grid_hd_tac_reg_error],
                #         feed_dict={net.x_vis: visual_batch, net.x_grid: grid_batch, net.x_hd: head_direction_batch, net.x_tac: tactile_batch, net.batch: j})

                # for k in range(cause_epoch):
                #     vis_cause, grid_cause, vis_grid_hd_tac_cause, vis_grid_hd_tac_reg_error = sess.run(
                #         [net.vis_update_cause, net.grid_update_cause, net.vis_grid_hd_tac_update_cause,
                #         net.vis_grid_hd_tac_reg_error],
                #         feed_dict={net.x_vis: visual_batch, net.x_grid: grid_batch, net.x_hd: head_direction_batch, net.x_tac: tactile_batch, net.batch: j})

                # for k in range(cause_epoch):
                #     vis_cause, vis_grid_hd_tac_reg_error = sess.run(
                #         [net.vis_update_cause,
                #         net.vis_grid_hd_tac_reg_error],
                #         feed_dict={net.x_vis: visual_batch, net.x_grid: grid_batch, net.x_hd: head_direction_batch, net.x_tac: tactile_batch, net.batch: j})

                # for k in range(cause_epoch):
                #     grid_cause, vis_grid_hd_tac_reg_error = sess.run(
                #         [net.grid_update_cause,
                #         net.vis_grid_hd_tac_reg_error],
                #         feed_dict={net.x_vis: visual_batch, net.x_grid: grid_batch, net.x_hd: head_direction_batch, net.x_tac: tactile_batch, net.batch: j})

                # for k in range(cause_epoch):
                #     vis_grid_hd_tac_cause, vis_grid_hd_tac_reg_error = sess.run(
                #         [net.vis_grid_hd_tac_update_cause,
                #         net.vis_grid_hd_tac_reg_error],
                #         feed_dict={net.x_vis: visual_batch, net.x_grid: grid_batch, net.x_hd: head_direction_batch, net.x_tac: tactile_batch, net.batch: j})

                # (optional) save reconstructions to diagnose training issues
                vis_reconstruction, grid_reconstruction, vis_grid_hd_tac_reconstruction = sess.run([net.vis_predictions[0], net.grid_predictions[0], net.vis_grid_hd_tac_predictions[0]],
                                                                feed_dict={ net.x_vis: visual_batch, 
                                                                            net.x_grid_0: grid_batch[:, 0, :], 
                                                                            net.x_grid_1: grid_batch[:, 1, :],
                                                                            net.x_grid_2: grid_batch[:, 2, :],
                                                                            net.x_grid_3: grid_batch[:, 3, :],
                                                                            net.x_grid_4: grid_batch[:, 4, :],
                                                                            net.x_hd: head_direction_batch, 
                                                                            net.x_tac: tactile_batch, 
                                                                            net.batch: j})

                if save_vis_causes:

                    np.save(cause_save_path + '/vis/epoch{}_batch{}_cause_0'.format(i, j), vis_cause[0])
                    np.save(cause_save_path + '/vis/epoch{}_batch{}_cause_1'.format(i, j), vis_cause[1])

                if save_grid_causes:

                    np.save(cause_save_path + '/grid/epoch{}_batch{}_cause_0'.format(i, j), grid_cause[0])

                if save_vis_grid_hd_tac_causes:

                    np.save(cause_save_path + '/vis_grid_hd_tac/epoch{}_batch{}'.format(i, j), vis_grid_hd_tac_cause[0])

                if save_vis_reconstructions:

                    np.save(reconstruction_save_path + '/vis/epoch{}_batch{}_reconstruction.npy'.format(i, j), vis_reconstruction)

                if save_grid_reconstructions:

                    np.save(reconstruction_save_path + '/grid/epoch{}_batch{}_reconstruction.npy'.format(i, j), grid_reconstruction)

                if save_vis_grid_hd_tac_reconstructions:

                    np.save(reconstruction_save_path + '/vis_grid_hd_tac/epoch{}_batch{}_reconstruction.npy'.format(i, j), vis_grid_hd_tac_reconstruction)

                # update weights
                _, _, _, vis_error, grid_error, vis_grid_hd_tac_error, vis_filter, grid_filter, vis_grid_hd_tac_filter = sess.run(
                    [net.vis_update_filter, net.grid_update_filter, net.vis_grid_hd_tac_update_filter,
                     net.vis_bu_error, net.grid_bu_error, net.vis_grid_hd_tac_bu_error,
                     net.vis_filters, net.grid_filters, net.vis_grid_hd_tac_filters],
                    feed_dict={ net.x_vis: visual_batch, 
                                net.x_grid_0: grid_batch[:, 0, :], 
                                net.x_grid_1: grid_batch[:, 1, :],
                                net.x_grid_2: grid_batch[:, 2, :],
                                net.x_grid_3: grid_batch[:, 3, :],
                                net.x_grid_4: grid_batch[:, 4, :],
                                net.x_hd: head_direction_batch, 
                                net.x_tac: tactile_batch, 
                                net.batch: j})

                # record maximum reconstruction error on the entire data
                vis_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > vis_epoch_loss[current_epoch, l]
                                                   else vis_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(vis_error)]
                grid_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > grid_epoch_loss[current_epoch, l]
                                                   else grid_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(grid_error)]
                vis_grid_hd_tac_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                    if np.max(np.mean(item, axis=1)) > vis_grid_hd_tac_epoch_loss[current_epoch, l]
                                                    else vis_grid_hd_tac_epoch_loss[current_epoch, l]
                                                    for l, item in enumerate(vis_grid_hd_tac_error)]

            # track average activity in inferred causes
            vis_avg_activity[current_epoch, :] = [np.mean(item) for item in vis_cause]
            grid_avg_activity[current_epoch, :] = [np.mean(item) for item in grid_cause]
            vis_grid_hd_tac_avg_activity[current_epoch, :] = [np.mean(item) for item in vis_grid_hd_tac_cause]

            print('-------- Epoch %d/%d --------\nVis Loss:%s Vis Mean Cause:%s\nGrid:%s Grid Mean Cause:%s\nvis_grid_hd_tac:%s vis_grid_hd_tac Mean Cause:%s' % (
                i+1, 
                n_epoch, 
                ', '.join(['%.8f' % elem for elem in vis_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in grid_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in grid_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_grid_hd_tac_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_grid_hd_tac_avg_activity[current_epoch, :]])))

        # create the save path if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save model and stats
        saver.save(sess, '%s/main.ckpt' % save_path)
        np.save('%s/vis_epoch_loss.npy' % save_path, vis_epoch_loss)
        np.save('%s/grid_epoch_loss.npy' % save_path, grid_epoch_loss)
        np.save('%s/vis_grid_hd_tac_epoch_loss.npy' % save_path, vis_grid_hd_tac_epoch_loss)
        np.save('%s/vis_avg_activity.npy' % save_path, vis_avg_activity)
        np.save('%s/grid_avg_activity.npy' % save_path, grid_avg_activity)
        np.save('%s/vis_grid_hd_tac_avg_activity.npy' % save_path, vis_grid_hd_tac_avg_activity)


if __name__ == '__main__':
    starttime = time.time()
    train()
    endtime = time.time()

    print ('Time taken: %f' % ((endtime - starttime) / 3600))

model_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/guifen_experiment_reconstruction/real_data_Gc5'

test_sample_start = sample_end

test_sample_end = test_sample_start + 50

num_test_samps = test_sample_end - test_sample_start

error_criterion = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])

max_inference_iteration= 500

class InferenceNetwork:
    def __init__(self, inp_shape, layers, cause_init, reg_causes, lr_causes, available_modalities = []):

        # create placeholders
        self.x_vis = tf.placeholder(tf.float32, shape=[1, inp_shape['vis']])
        self.x_grid_0 = tf.placeholder(tf.float32, shape=[1, inp_shape['grid']])
        self.x_grid_1 = tf.placeholder(tf.float32, shape=[1, inp_shape['grid']])
        self.x_grid_2 = tf.placeholder(tf.float32, shape=[1, inp_shape['grid']])
        self.x_grid_3 = tf.placeholder(tf.float32, shape=[1, inp_shape['grid']])
        self.x_grid_4 = tf.placeholder(tf.float32, shape=[1, inp_shape['grid']])
        self.x_hd = tf.placeholder(tf.float32, shape=[1, inp_shape['hd']])
        self.x_tac = tf.placeholder(tf.float32, shape=[1, inp_shape['tac']])

        # create filters and cause for visual data
        self.vis_filters = []
        self.vis_causes = []
        for i in range(len(layers['vis'])):
            filter_name = 'vis_filter_%d' % i
            cause_name = 'vis_cause_%d' % i

            if i == 0:
                self.vis_filters += [tf.get_variable(filter_name, shape=[layers['vis'][i], inp_shape['vis']])]
            else:
                self.vis_filters += [tf.get_variable(filter_name, shape=[layers['vis'][i], layers['vis'][i-1]])]

            init = tf.constant_initializer(cause_init['vis'][i])
            self.vis_causes += [tf.get_variable(cause_name, shape=[1, layers['vis'][i]], initializer=init)]

         # create filters and causes for grid cells
        self.grid_filters = []
        self.grid_causes = []
        for i in range(len(layers['grid'])):
            cause_name = 'grid_cause_%d' % i

            if i == 0:
                filter_name = 'grid_filter_0_0'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_1'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_2'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_3'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
                filter_name = 'grid_filter_0_4'
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
            else:
                filter_name = 'grid_filter_%d' % i
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], layers['grid'][i-1]])]

            init = tf.constant_initializer(cause_init['grid'][i])
            self.grid_causes += [tf.get_variable(cause_name, shape=[1, layers['grid'][i]], initializer=init)]

        # create filters and cause for latent space visual + grid + head direction + tactile
        self.vis_grid_hd_tac_filters = []
        self.vis_grid_hd_tac_causes = []
        for i in range(len(layers['vis_grid_hd_tac'])):
            if i == 0:
                # add filters for vis
                filter_name = 'vis_grid_hd_tac->vis_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   layers['vis'][-1]])]

                # add filters for grid
                filter_name = 'vis_grid_hd_tac->grid_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   layers['grid'][-1]])]

                # add filters for hd
                filter_name = 'vis_grid_hd_tac->hd_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   inp_shape['hd']])]

                # add filters for grid
                filter_name = 'vis_grid_hd_tac->tac_filter'
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   inp_shape['tac']])]
                
            else:
                filter_name = 'vis_grid_hd_tac_filter_%d' % i
                self.vis_grid_hd_tac_filters += [tf.get_variable(filter_name, shape=[layers['vis_grid_hd_tac'][i],
                                                                                   layers['vis_grid_hd_tac'][i - 1]])]

            cause_name = 'vis_grid_hd_tac_causes_%d' % i
            init = tf.constant_initializer(cause_init['vis_grid_hd_tac'][i])
            self.vis_grid_hd_tac_causes += [tf.get_variable(cause_name, shape=[1, layers['vis_grid_hd_tac'][i]], initializer=init)]

        # compute predictions

        # vis predictions
        self.vis_predictions = []
        for i in range(len(layers['vis'])):
            self.vis_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_causes[i], self.vis_filters[i]))]

        # grid predictions
        self.grid_predictions = []
        for i in range(len(layers['grid'])):
            if i == 0:
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i+1]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i+2]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i+3]))]
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i+4]))]
            else:
                self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i+4]))]

        # vis_grid_hd_tac predictions
        self.vis_grid_hd_tac_predictions = []
        for i in range(len(layers['vis_grid_hd_tac'])):
            if i == 0:
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_causes[i], self.vis_grid_hd_tac_filters[i]))] # vis prediction
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_causes[i], self.vis_grid_hd_tac_filters[i+1]))] # grid prediction
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_causes[i], self.vis_grid_hd_tac_filters[i+2]))] # hd prediction
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_causes[i], self.vis_grid_hd_tac_filters[i+3]))] # tac prediction
            else:
                self.vis_grid_hd_tac_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_grid_hd_tac_causes[i], self.vis_grid_hd_tac_filters[i+3]))]

        # add ops for computing gradients for vis causes
        self.vis_bu_error = []
        self.vis_update_filter = []
        self.vis_cause_grad = []
        for i in range(len(layers['vis'])):
            if i == 0:
                self.vis_bu_error += [tf.losses.mean_squared_error(self.x_vis, self.vis_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.vis_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_causes[i - 1]), self.vis_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(layers['vis']) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_predictions[i+1]), self.vis_causes[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_grid_hd_tac_predictions[0]), self.vis_causes[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['vis'][i] * (self.vis_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_causes[i])
            self.vis_cause_grad += [tf.gradients([self.vis_bu_error[i], td_error, reg_error],
                                                          self.vis_causes[i])[0]]

        # add ops for computing gradients for grid causes
        self.grid_bu_error = []
        self.grid_update_filter = []
        self.grid_cause_grad = []
        for i in range(len(layers['grid'])):
            if i == 0:
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_0, self.grid_predictions[0],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_1, self.grid_predictions[1],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_2, self.grid_predictions[2],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_3, self.grid_predictions[3],
                                                                            reduction=tf.losses.Reduction.NONE)]
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid_4, self.grid_predictions[4],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.grid_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_causes[i - 1]), self.grid_predictions[i+4],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(layers['grid']) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_predictions[i+4]), self.grid_causes[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_grid_hd_tac_predictions[1]), self.grid_causes[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['grid'][i] * (self.grid_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_causes[i])
            error_list = [  self.grid_bu_error[i], self.grid_bu_error[i+1], self.grid_bu_error[i+2],
                            self.grid_bu_error[i+3], self.grid_bu_error[i+4], td_error, reg_error]
            if i == 0:
                error_list = [  self.grid_bu_error[i], self.grid_bu_error[i+1], self.grid_bu_error[i+2],
                                self.grid_bu_error[i+3], self.grid_bu_error[i+4], td_error, reg_error]
                assert len(error_list) == len(self.grid_bu_error) + 2
            else:
                error_list = [  self.grid_bu_error[i+4], td_error, reg_error]
                assert len(error_list) == 3
            self.grid_cause_grad += [tf.gradients(error_list, self.grid_causes[i])[0]]

        # add ops for computing gradients for vis_grid_hd_tac causes
        self.vis_grid_hd_tac_bu_error = []
        self.vis_grid_hd_tac_reg_error = []
        self.vis_grid_hd_tac_update_filter = []
        self.vis_grid_hd_tac_cause_grad = []
        for i in range(len(layers['vis_grid_hd_tac'])):
            if i == 0:
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_causes[-1]), self.vis_grid_hd_tac_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_causes[-1]), self.vis_grid_hd_tac_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_hd), self.vis_grid_hd_tac_predictions[i+2],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_grid_hd_tac_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_tac), self.vis_grid_hd_tac_predictions[i+3],
                    reduction=tf.losses.Reduction.NONE)]

                self.vis_grid_hd_tac_reg_error += [reg_causes['vis_grid_hd_tac'][i] * (self.vis_grid_hd_tac_causes[i] ** 2)]
                # self.vis_hd_reg_error += [tf.keras.regularizers.l2(reg_causes['vis_hd'][i])(self.vis_hd_minibatch[i])]
                if len(layers['vis_grid_hd_tac']) > 1:
                    raise NotImplementedError
                else:
                    error_list = [self.vis_grid_hd_tac_reg_error]

                    if 'visual' in available_modalities:
                        error_list += [self.vis_grid_hd_tac_bu_error[0]]

                    if 'grid' in available_modalities:
                        error_list += [self.vis_grid_hd_tac_bu_error[1]]

                    if 'hd' in available_modalities:
                        error_list += [self.vis_grid_hd_tac_bu_error[2]]

                    if 'tac' in available_modalities:
                        error_list += [self.vis_grid_hd_tac_bu_error[3]]

                    self.vis_grid_hd_tac_cause_grad += [
                        tf.gradients(error_list, self.vis_grid_hd_tac_causes[i])[0]]
            else:
                raise NotImplementedError

        # add ops for updating causes
        self.vis_update_cause = []
        self.hd_update_cause = []
        self.grid_update_cause = []
        self.grid_hd_update_cause = []
        self.vis_hd_update_cause = []
        self.vis_grid_hd_tac_update_cause = []
        with tf.control_dependencies(self.vis_cause_grad + self.grid_cause_grad + self.vis_grid_hd_tac_cause_grad):
            # vis modality
            for i in range(len(layers['vis'])):
                self.vis_update_cause += [tf.assign_sub(self.vis_causes[i], (lr_causes['vis'][i] * self.vis_cause_grad[i]))]

            # grid modality
            for i in range(len(layers['grid'])):
                self.grid_update_cause += [tf.assign_sub(self.grid_causes[i], (lr_causes['grid'][i] * self.grid_cause_grad[i]))]

            # vis_grid_hd_tac modality
            for i in range(len(layers['vis_grid_hd_tac'])):
                self.vis_grid_hd_tac_update_cause += [tf.assign_sub(self.vis_grid_hd_tac_causes[i], (lr_causes['vis_grid_hd_tac'][i] * self.vis_grid_hd_tac_cause_grad[i]))]

def init_network(model_path, available_modalities = []):
    tf.reset_default_graph()

    net = InferenceNetwork(inp_shape, layers, cause_init, reg_causes, lr_causes, available_modalities)

    saver = tf.compat.v1.train.Saver(net.vis_filters + net.grid_filters + net.vis_grid_hd_tac_filters)
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, '%s/main.ckpt' % model_path)

    return sess, net

def infer_repr(sess, net, sample_index, max_inference_iteration=1, error_criterion=0.0001, visual_data=None, grid_data = None, head_direction_data=None, tactile_data = None, verbose=False,
               available_modalities = []):
    inference_iteration = 1

    while True:
        # infer representations
        vis_cause, grid_cause, vis_grid_hd_tac_cause, vis_error, grid_error, vis_grid_hd_tac_error, vis_pred, grid_pred, vis_grid_hd_tac_pred, vis_filter, grid_filter, vis_grid_hd_tac_filter = sess.run(
                [net.vis_update_cause, net.grid_update_cause, net.vis_grid_hd_tac_update_cause,
                 net.vis_bu_error, net.grid_bu_error, net.vis_grid_hd_tac_bu_error,
                 net.vis_predictions, net.grid_predictions, net.vis_grid_hd_tac_predictions,
                 net.vis_filters, net.grid_filters, net.vis_grid_hd_tac_filters],
                 feed_dict={    net.x_vis: visual_data, 
                                net.x_grid_0: grid_data[None, 0, 0, :], 
                                net.x_grid_1: grid_data[None, 0, 1, :], 
                                net.x_grid_2: grid_data[None, 0, 2, :], 
                                net.x_grid_3: grid_data[None, 0, 3, :], 
                                net.x_grid_4: grid_data[None, 0, 4, :], 
                                net.x_hd: head_direction_data, 
                                net.x_tac: tactile_data})

        if 'visual' in available_modalities:
            vis_epoch_loss = [np.mean(item) for item in vis_error]
        else:
            vis_epoch_loss = [np.NINF for item in vis_error]

        if 'grid' in available_modalities:
            grid_epoch_loss = [np.mean(item) for item in grid_error]
        else:
            grid_epoch_loss = [np.NINF for item in grid_error]

        if 'hd' in available_modalities:
            hd_epoch_loss = [np.mean(item) for item in vis_grid_hd_tac_error[2]]
        else:
            hd_epoch_loss = [np.NINF]

        if 'tac' in available_modalities:
            tac_epoch_loss = [np.mean(item) for item in vis_grid_hd_tac_error[3]]
        else:
            tac_epoch_loss = [np.NINF]

        if (np.all(np.array(vis_epoch_loss + grid_epoch_loss + hd_epoch_loss + tac_epoch_loss) < error_criterion)) or (inference_iteration >= max_inference_iteration):
            if verbose:
                print_str = ', '.join(['%.4f' % elem for elem in vis_epoch_loss + grid_epoch_loss + hd_epoch_loss + tac_epoch_loss])
                print ('Test Sample %d: (%d/%d) %s' % (sample_index, inference_iteration, max_inference_iteration, print_str))
            break
        else:
            inference_iteration += 1

    # reconstruct the missing modality
    recon_vis = np.dot(vis_grid_hd_tac_cause[0], vis_grid_hd_tac_filter[0])
    for l in range(len(vis_filter), 0, -1):
        recon_vis = np.dot(recon_vis, vis_filter[l - 1])
    recon_grid = np.dot(vis_grid_hd_tac_cause[0], vis_grid_hd_tac_filter[1])
    for l in range(len(grid_filter), 0, -1):
        if l > 5: # If more than number of 'splits' at level 0
            recon_grid = np.dot(recon_grid, grid_filter[l - 1])
        else:
            recon_grid_0 = np.dot(recon_grid, grid_filter[0])
            recon_grid_1 = np.dot(recon_grid, grid_filter[1])
            recon_grid_2 = np.dot(recon_grid, grid_filter[2])
            recon_grid_3 = np.dot(recon_grid, grid_filter[3])
            recon_grid_4 = np.dot(recon_grid, grid_filter[4])
    recon_head_direction = np.dot(vis_grid_hd_tac_cause[0], vis_grid_hd_tac_filter[2])
    recon_tac = np.dot(vis_grid_hd_tac_cause[0], vis_grid_hd_tac_filter[3])

    return vis_grid_hd_tac_cause, recon_vis, recon_grid_0, recon_grid_1, recon_grid_2, recon_grid_3, recon_grid_4, recon_head_direction, recon_tac

def run_inference(sess, net, visual_data, grid_data, head_direction_data, tactile_data, representations_save_path, available_modalities = []):

    print("Modalities: {}, Inferences will be saved to: {}".format(",".join(available_modalities), representations_save_path))

    print(visual_data.shape)
    print(grid_data.shape)
    print(head_direction_data.shape)
    print(tactile_data.shape)

    assert visual_data.shape[0] == head_direction_data.shape[0]
    assert visual_data.shape[0] == grid_data.shape[0]
    assert visual_data.shape[0] == tactile_data.shape[0]

    recon_temp_vis = np.zeros([ visual_data.shape[0], 45, 80, 3 ])
    # recon_temp_grid_0 = np.zeros([ grid_data.shape[0], 1000 ])
    # recon_temp_grid_1 = np.zeros([ grid_data.shape[0], 1000 ])
    # recon_temp_grid_2 = np.zeros([ grid_data.shape[0], 1000 ])
    # recon_temp_grid_3 = np.zeros([ grid_data.shape[0], 1000 ])
    # recon_temp_grid_4 = np.zeros([ grid_data.shape[0], 1000 ])
    recon_temp_grid = np.zeros([ grid_data.shape[0], 5, 1000 ])
    recon_temp_hd = np.zeros([ head_direction_data.shape[0], 180 ])
    recon_temp_tac = np.zeros([ tactile_data.shape[0], 48 ])
    representations = np.zeros([visual_data.shape[0], 300])

    sess.run(tf.compat.v1.variables_initializer(net.vis_causes + net.grid_causes + net.vis_grid_hd_tac_causes))

    for j in range (visual_data.shape[0]):

        visual_input = visual_data[None, j]
        grid_input = grid_data[None, j]
        head_direction_input = head_direction_data[None, j]
        tactile_input = tactile_data[None, j]

        if 'visual' not in available_modalities:
            visual_input = np.zeros([1, 10800])
            
        if 'grid' not in available_modalities:
            grid_input = np.zeros([1, 5, 1000])

        if 'hd' not in available_modalities:
            head_direction_input = np.zeros([1, 180])

        if 'tac' not in available_modalities:
            tactile_input = np.zeros([1, 48])

        reps, recon_vis, recon_grid_0, recon_grid_1, recon_grid_2, recon_grid_3, recon_grid_4, recon_hd, recon_tac = infer_repr(sess, net, j+1, max_inference_iteration, error_criterion, visual_input, grid_input, head_direction_input, tactile_input, True, available_modalities)
        representations[j, :] = reps[0]
        recon_temp_vis[j, :] = recon_vis.reshape(45,80,3) # reform into image
        recon_temp_grid[j, :] = np.concatenate([recon_grid_0, recon_grid_1, recon_grid_2, recon_grid_3, recon_grid_4])
        recon_temp_hd[j, :] = recon_hd
        recon_temp_tac[j, :] = recon_tac
        

    print('Done!')

    np.save(representations_save_path + 'representations.npy', representations)
    sio.savemat(representations_save_path + 'reps.mat', {'reps':representations})

    np.save(representations_save_path + 'reconstructions_visual.npy', recon_temp_vis)
    sio.savemat(representations_save_path + 'recon_vis.mat', {'recon_vis':recon_temp_vis})

    np.save(representations_save_path + 'reconstructions_grid_cells.npy', recon_temp_grid)
    sio.savemat(representations_save_path + 'recon_grid.mat', {'recon_grid':recon_temp_grid})

    np.save(representations_save_path + 'reconstructions_hd_cells.npy', recon_temp_hd)
    sio.savemat(representations_save_path + 'recon_hd.mat', {'recon_hd':recon_temp_hd})

    np.save(representations_save_path + 'reconstructions_tactile.npy', recon_temp_tac)
    sio.savemat(representations_save_path + 'recon_tactile.mat', {'recon_tactile':recon_temp_tac})

def generate_representations(shuffle):

    root_test_path = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets"
    root_representations_path = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_"

    #dataset = 1
    if True:
        dataset = 'real_data_6005_timestamped'
    #if False:
    #for dataset in (1,6):#,10,11,12):#13,14,15,16,17,18,19,20):#for dataset in (9,10,11,12,1,2,3,4,5,6,7,8):

        test_set = root_test_path + "/{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_baseline/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}_Gc5/all/".format(dataset)

        visual_data, grid_data, head_direction_data, tactile_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modalities = ['visual', 'grid', 'hd', 'tac'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, grid_data, head_direction_data, tactile_data, representations_save_path, available_modalities = ['visual', 'grid', 'hd', 'tac'])

    if True:
        dataset = 'real_data_6005_timestamped'
    #if False:
    #for dataset in (1,6):#,10,11,12):#for dataset in (1,11):
    #for dataset in ("circling_proximal", "random_distal", "random_proximal", "cogarch"): #("rotating_distal", "rotating_proximal", "circling_distal", "circling_proximal", "random_distal", "random_proximal", "cogarch"):
    #for dataset in ("random_distal_2", "random_distal_3", "random_distal_4", "random_distal_5"):

        test_set = root_test_path + "/{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_baseline/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}_Gc5/no_grid/".format(dataset)

        visual_data, grid_data, head_direction_data, tactile_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modalities = ['visual', 'hd', 'tac'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, grid_data, head_direction_data, tactile_data, representations_save_path, available_modalities = ['visual', 'hd', 'tac'])

    if True:
        dataset = 'real_data_6005_timestamped'
    #if False:
    #for dataset in (1,6):#,10,11,12):#for dataset in (9,10,11,12,1,2,3,4,5,6,7,8):

        test_set = root_test_path + "/{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_baseline/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}_Gc5/no_hd/".format(dataset)

        visual_data, grid_data, head_direction_data, tactile_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modalities = ['visual', 'grid', 'tac'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, grid_data, head_direction_data, tactile_data, representations_save_path, available_modalities = ['visual', 'grid', 'tac'])

    #if True:
    if False:
    #for dataset in (1,6):#,10,11,12):#for dataset in (9,10,11,12,1,2,3,4,5,6,7,8):

        test_set = root_test_path + "{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_numb_blackout/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}/grid_only/".format(dataset)

        visual_data, head_direction_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modality = ['grid'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, head_direction_data, representations_save_path, available_modality = ['grid'])

starttime = time.time()

generate_representations(shuffle = False)

endtime = time.time()

print ('Time taken: %f' % ((endtime - starttime) / 3600))

# Create timestamps file for proper injection of corrections into SNN models

#body_head_direction = np.load(data_path + '/body_pose.npy')[:,0]

#np.save(data_path + '/training_timestamps.npy', body_head_direction[sample_start:sample_end])

#np.save(data_path + '/test_timestamps.npy', body_head_direction[test_sample_start:test_sample_end])