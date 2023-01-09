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

    head_direction_file = '/networkOutput.npy'

    #tactile_file = '/whisker_data.npy'
    tactile_file = '/whisker_data.csv'

    visual_data = np.load(data_path + images_file)[sample_start:sample_end]

    visual_data = visual_data.reshape(visual_data.shape[0], 10800)/255 # flatten

    head_direction_data = np.load(data_path + head_direction_file).T[sample_start:sample_end]#np.load(data_path + '/networkOutput.npy').T

    grid_data = np.load(data_path + grid_file)[sample_start:sample_end*10:10,:] # Grid data is at 50Hz, so must be subsampled

    grid_data = grid_data.reshape(grid_data.shape[0], 5000)

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

def circular_mean(causes, gaussian = False):

    causes_length = causes.get_shape().as_list()[1]

    arc_length = (2*np.pi)/causes_length

    angles_in_radians = np.arange(causes_length) * arc_length

    cos_mean = np.cos(angles_in_radians)

    sin_mean = np.sin(angles_in_radians)

    circular_mean = np.arctan2(sin_mean, cos_mean)

    if gaussian is True:

        gaussian_range = np.arange(-(causes_length//2),(causes_length//2))

        gaussian_function = norm(circular_mean, causes_length//10)

        gaussian_mean = gaussian_function.pdf(gaussian_range)

        circular_mean = minmax_scale(gaussian_mean)

    return circular_mean

### User-defined Parameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

sample_start = 0

sample_end = 3000

n_sample = sample_end - sample_start        # If you have collected your own dataset, you will need to determine how many samples where collected in the run
                                            # Alternatively, if you are using a built-in dataset, copy the sample number as described in the datasets' README

minibatch_sz = 10                            # Minibatch size. Can be left as default for physical data, for simulated data good numbers to try are 40, 50 and 100
                                            # Datasize size must be fully divisible by minibatch size

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_grid_cells_test_dataset'               # Path to training data. Training data should be in .npy format:

save_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/visual_head_direction_grid_cells/stacked'#trial1'               # Path to save trained model to (once trained)
load_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/visual_head_direction_grid_cells/stacked'#trial1'               # Path to load trained model from (after training, or if already trained beforehand)

cause_save_path = save_path + '/causes'#trial1/causes'  # Path to save causes to (optional, for training diagnostics)
reconstruction_save_path = save_path + '/reconstructions'#trial1/reconstructions'  # Path to save reconstructons to (optional, for training diagnostics)

save_vis_causes = False
save_hd_causes = False
save_vis_hd_causes = False
save_grid_causes = False
save_vis_hd_grid_causes = False

save_vis_reconstructions = False
save_vis_hd_reconstructions = False
save_grid_reconstructions = True
save_vis_hd_grid_reconstructions = False

n_epoch = 200                               # Number of training epochs to generate model. Default is 200
                                            
shuffle_data = False                        # Do you want to shuffle the training data? Default is False

# Load the data from .mat files

#visual_data, head_direction_data = load_mat_data(data_path, shuffle_data)

# Alternatively, load the data from .npy files

visual_data, head_direction_data, grid_data = load_npy_data(data_path, sample_start, sample_end, minibatch_sz, shuffle_data)

### Model Hyperparameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

load_model = False                          # If True, load a previously trained model from load_path. If False, train from scratch.

inp_shape =    {"vis": visual_data.shape[1],
                "hd": head_direction_data.shape[1],
                "grid": grid_data.shape[1]}

layers =   {"vis": [1000, 300],
            "hd": [120, 60],
            "vis_hd": [100],
            "grid": [200],
            "vis_hd_grid": [100]}

cause_init =   {"vis": [0.25, 0.25],
                "hd": [0.25, 0.25],
                "vis_hd": [0.25],
                "grid": [0.25],
                "vis_hd_grid": [0.25]}

reg_causes =   {"vis": [0.0, 0.0],
                "hd": [0.2, 0.2],
                "vis_hd": [0.0],
                "grid": [0.0],
                "vis_hd_grid": [0.0]}

lr_causes =    {"vis": [0.0004, 0.0004],
                "hd": [0.0004, 0.0004],
                "vis_hd": [0.0004],
                "grid": [0.0004, 0.0004],
                "vis_hd_grid": [0.0004]}

reg_filters =  {"vis": [0.0, 0.0],
                "hd": [0.2, 0.2],
                "vis_hd": [0.0, 0.0],
                "grid": [0.0, 0.0],
                "vis_hd_grid": [0.0, 0.0]}

lr_filters =   {"vis": [0.0001, 0.0001],
                "hd": [0.00001, 0.00001],
                "vis_hd": [0.0001, 0.0001],
                "grid": [0.0001, 0.0001],
                "vis_hd_grid": [0.0001, 0.0001]}

class Network:
    def __init__(self, n_sample, minibatch_sz, inp_shape, layers, cause_init, reg_causes, lr_causes, reg_filters, lr_filters):

        # create placeholders
        self.x_vis = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['vis']])
        self.x_hd = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['hd']])
        self.x_grid = tf.placeholder(tf.float32, shape=[minibatch_sz, inp_shape['grid']])
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

        # create filters and causes for head direction data
        self.hd_filters = []
        self.hd_causes = []
        for i in range(len(layers['hd'])):
           filter_name = 'hd_filter_%d' % i
           cause_name = 'hd_cause_%d' % i

           if i == 0:
               self.hd_filters += [tf.get_variable(filter_name, shape=[layers['hd'][i], inp_shape['hd']])]
           else:
               self.hd_filters += [tf.get_variable(filter_name, shape=[layers['hd'][i], layers['hd'][i-1]])]

           init = tf.constant_initializer(cause_init['hd'][i])
           self.hd_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['hd'][i]], initializer=init)]

        # create filters and causes for grid cells
        self.grid_filters = []
        self.grid_causes = []
        for i in range(len(layers['grid'])):
            filter_name = 'grid_filter_%d' % i
            cause_name = 'grid_cause_%d' % i

            if i == 0:
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
            else:
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], layers['grid'][i-1]])]

            init = tf.constant_initializer(cause_init['grid'][i])
            self.grid_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['grid'][i]], initializer=init)]

        # create filters and causes for visual and head direction latent space
        self.vis_hd_filters = []
        self.vis_hd_causes = []
        for i in range(len(layers['vis_hd'])):
            if i == 0:
                # add filters for vis
                filter_name = 'vis_hd->vis_filter'
                self.vis_hd_filters += [tf.get_variable(filter_name, shape=[layers['vis_hd'][i],
                                                                                   layers['vis'][-1]])]
                # add filters for hd
                filter_name = 'vis_hd->hd_filter'
                self.vis_hd_filters += [tf.get_variable(filter_name, shape=[layers['vis_hd'][i],
                                                                                   inp_shape['hd']])]
            else:
                filter_name = 'vis_hd_filter_%d' % i
                self.vis_hd_filters += [tf.get_variable(filter_name, shape=[layers['vis_hd'][i],
                                                                                   layers['vis_hd'][i - 1]])]

            cause_name = 'vis_hd_cause_%d' % i
            init = tf.constant_initializer(cause_init['vis_hd'][i])
            self.vis_hd_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['vis_hd'][i]], initializer=init)]

        # create filters and causes for grid and head direction latent space
        self.grid_hd_filters = []
        self.grid_hd_causes = []
        for i in range(len(layers['grid_hd'])):
            if i == 0:
                # add filters for grid
                filter_name = 'grid_hd->grid_filter'
                self.grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['grid_hd'][i],
                                                                                   layers['grid'][-1]])]
                # add filters for vis_hd
                filter_name = 'grid_hd->hd_filter'
                self.grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['grid_hd'][i],
                                                                                   layers['hd'][-1]])]
            else:
                filter_name = 'grid_hd_filter_%d' % i
                self.grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['grid_hd'][i],
                                                                                   layers['grid_hd'][i - 1]])]

            cause_name = 'grid_hd_causes_%d' % i
            init = tf.constant_initializer(cause_init['grid_hd'][i])
            self.grid_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['grid_hd'][i]], initializer=init)]

        # create filters and cause for 2nd tier latent space (visual+head_direction) + (grid+head_direction)
        self.vis_hd_grid_hd_filters = []
        self.vis_hd_grid_hd_causes = []
        for i in range(len(layers['(vis_hd)_(grid_hd)'])):
            if i == 0:
                # add filters for grid
                filter_name = '(vis_hd)_(grid_hd)->grid_hd_filter'
                self.vis_hd_grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['(vis_hd)_(grid_hd)'][i],
                                                                                   layers['grid_hd'][-1]])]
                # add filters for vis_hd
                filter_name = '(vis_hd)_(grid_hd)->vis_hd_filter'
                self.vis_hd_grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['(vis_hd)_(grid_hd)'][i],
                                                                                   layers['vis_hd'][-1]])]
            else:
                filter_name = '(vis_hd)_(grid_hd)_filter_%d' % i
                self.vis_hd_grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['(vis_hd)_(grid_hd)'][i],
                                                                                   layers['(vis_hd)_(grid_hd)'][i - 1]])]

            cause_name = '(vis_hd)_(grid_hd)_causes_%d' % i
            init = tf.constant_initializer(cause_init['(vis_hd)_(grid_hd)'][i])
            self.vis_hd_grid_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['(vis_hd)_(grid_hd)'][i]], initializer=init)]

        # compute predictions
        current_batch = tf.range(self.batch * minibatch_sz, (self.batch + 1) * minibatch_sz)
        # vis predictions
        self.vis_minibatch = []
        self.vis_predictions = []
        for i in range(len(layers['vis'])):
            self.vis_minibatch += [tf.gather(self.vis_causes[i], indices=current_batch, axis=0)]
            self.vis_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_minibatch[i], self.vis_filters[i]))]

        # hd predictions
        # self.hd_minibatch = []
        # self.hd_predictions = []
        # for i in range(len(layers['hd'])):
        #     self.hd_minibatch += [tf.gather(self.hd_causes[i], indices=current_batch, axis=0)]
        #     self.hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.hd_minibatch[i], self.hd_filters[i]))]

        # grid predictions
        self.grid_minibatch = []
        self.grid_predictions = []
        for i in range(len(layers['grid'])):
            self.grid_minibatch += [tf.gather(self.grid_causes[i], indices=current_batch, axis=0)]
            self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_minibatch[i], self.grid_filters[i]))]

        # vis_hd predictions
        self.vis_hd_minibatch = []
        self.vis_hd_predictions = []
        for i in range(len(layers['vis_hd'])):
            self.vis_hd_minibatch += [tf.gather(self.vis_hd_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.vis_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_minibatch[i], self.vis_hd_filters[i]))]  # vis prediction
                self.vis_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_minibatch[i], self.vis_hd_filters[i+1]))]  # hd prediction
            else:
                self.vis_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_minibatch[i], self.vis_hd_filters[i+1]))]

        # grid_hd predictions
        self.grid_hd_minibatch = []
        self.grid_hd_predictions = []
        for i in range(len(layers['vis_hd'])):
            self.grid_hd_minibatch += [tf.gather(self.grid_hd_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_hd_minibatch[i], self.grid_hd_filters[i]))]  # vis prediction
                self.grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_hd_minibatch[i], self.grid_hd_filters[i+1]))]  # hd prediction
            else:
                self.grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_hd_minibatch[i], self.grid_hd_filters[i+1]))]

        # (vis_hd)_(grid_hd) predictions
        self.vis_hd_grid_hd_minibatch = []
        self.vis_hd_grid_hd_predictions = []
        for i in range(len(layers['(vis_hd)_(grid_hd)'])):
            self.vis_hd_grid_hd_minibatch += [tf.gather(self.vis_hd_grid_hd_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.vis_hd_grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_grid_hd_minibatch[i], self.vis_hd_grid_hd_filters[i]))]  # vis prediction
                self.vis_hd_grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_grid_hd_minibatch[i], self.vis_hd_grid_hd_filters[i+1]))]  # hd prediction
            else:
                self.vis_hd_grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_grid_hd_minibatch[i], self.vis_hd_grid_hd_filters[i+1]))]

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
                    tf.stop_gradient(self.vis_hd_predictions[0]), self.vis_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['vis'][i] * (self.vis_minibatch[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_minibatch[i])
            self.vis_cause_grad += [tf.gradients([self.vis_bu_error[i], td_error, reg_error],
                                                          self.vis_minibatch[i])[0]]

            # ops for updating weights
            reg_error = reg_filters['vis'][i] * (self.vis_filters[i] ** 2)
            vis_filter_grad = tf.gradients([self.vis_bu_error[i], reg_error], self.vis_filters[i])[0]
            self.vis_update_filter += [
                tf.assign_sub(self.vis_filters[i], lr_filters['vis'][i] * vis_filter_grad)]

        # add ops for computing gradients for hd causes and for updating weights
        # self.hd_bu_error = []
        # self.hd_update_filter = []
        # self.hd_cause_grad = []
        # for i in range(len(layers['hd'])):
        #     if i == 0:
        #         self.hd_bu_error += [tf.losses.mean_squared_error(self.x_hd, self.hd_predictions[i],
        #                                                                     reduction=tf.losses.Reduction.NONE)]
        #     else:
        #         self.hd_bu_error += [tf.losses.mean_squared_error(
        #             tf.stop_gradient(self.hd_minibatch[i - 1]), self.hd_predictions[i],
        #             reduction=tf.losses.Reduction.NONE)]

        #     # compute top-down prediction error
        #     if len(layers['hd']) > (i + 1):
        #         # there are more layers in this modality
        #         td_error = tf.losses.mean_squared_error(
        #             tf.stop_gradient(self.hd_predictions[i+1]), self.hd_minibatch[i],
        #                 reduction=tf.losses.Reduction.NONE)
        #     else:
        #         # this is the only layer in this modality
        #         td_error = tf.losses.mean_squared_error(
        #             tf.stop_gradient(self.vis_hd_predictions[1]), self.hd_minibatch[i],
        #                 reduction=tf.losses.Reduction.NONE)

        #     #reg_error = reg_causes['hd'][i] * (self.hd_minibatch[i] ** 2)
        #     # reg_error = tf.keras.regularizers.l2(reg_causes['hd'][i])(self.hd_minibatch[i])
        #     reg_error = reg_causes['vis'][i] * ((self.vis_minibatch[i] - circular_mean(self.vis_minibatch[i], gaussian = False)) ** 2)
        #     self.hd_cause_grad += [
        #         tf.gradients([self.hd_bu_error[i], td_error, reg_error], self.hd_minibatch[i])[0]]

        #     # add ops for updating weights
        #     reg_error = reg_filters['hd'][i] * (self.hd_filters[i] ** 2)
        #     hd_filter_grad = tf.gradients([self.hd_bu_error[i], reg_error], self.hd_filters[i])[0]
        #     self.vis_update_filter += [
        #         tf.assign_sub(self.hd_filters[i], lr_hd_filters[i] * hd_filter_grad)]
        #     #else:
        #         #raise NotImplementedError

        # add ops for computing gradients for vis causes and for updating weights
        self.grid_bu_error = []
        self.grid_update_filter = []
        self.grid_cause_grad = []
        for i in range(len(layers['grid'])):
            if i == 0:
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid, self.grid_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.grid_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_minibatch[i - 1]), self.grid_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(layers['grid']) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_predictions[i+1]), self.grid_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_hd_grid_predictions[0]), self.grid_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['grid'][i] * (self.grid_minibatch[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_minibatch[i])
            self.grid_cause_grad += [tf.gradients([self.grid_bu_error[i], td_error, reg_error],
                                                          self.grid_minibatch[i])[0]]

            # ops for updating weights
            reg_error = reg_filters['grid'][i] * (self.grid_filters[i] ** 2)
            grid_filter_grad = tf.gradients([self.grid_bu_error[i], reg_error], self.grid_filters[i])[0]
            self.grid_update_filter += [
                tf.assign_sub(self.grid_filters[i], lr_filters['grid'][i] * grid_filter_grad)]

        # add ops for computing gradients for vis_hd causes
        self.vis_hd_bu_error = []
        self.vis_hd_reg_error = []
        self.vis_hd_update_filter = []
        self.vis_hd_cause_grad = []
        for i in range(len(layers['vis_hd'])):
            if i == 0:
                self.vis_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_minibatch[-1]), self.vis_hd_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_hd), self.vis_hd_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.vis_hd_reg_error += [reg_causes['vis_hd'][i] * (self.vis_hd_causes[i] ** 2)]
                # self.vis_hd_reg_error += [tf.keras.regularizers.l2(reg_causes['vis_hd'][i])(self.vis_hd_minibatch[i])]
                if len(layers['vis_hd']) > 1:
                    raise NotImplementedError
                else:
                    self.vis_hd_cause_grad += [
                        tf.gradients([self.vis_hd_bu_error[i], self.vis_hd_bu_error[i+1], self.vis_hd_reg_error[i]],
                                               self.vis_hd_minibatch[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['vis_hd'][i] * (self.vis_hd_filters[i] ** 2)
                vis_hd_filter_grad = tf.gradients([self.vis_hd_bu_error[i], reg_error], self.vis_hd_filters[i])[0]
                self.vis_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_filters[i], lr_filters['vis_hd'][i] * vis_hd_filter_grad)]
                reg_error = reg_filters['vis_hd'][i+1] * (self.vis_hd_filters[i+1] ** 2)
                vis_hd_filter_grad = tf.gradients([self.vis_hd_bu_error[i+1], reg_error], self.vis_hd_filters[i+1])[0]
                self.vis_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_filters[i+1], lr_filters['vis_hd'][i+1] * vis_hd_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for computing gradients for grid_hd causes
        self.grid_hd_bu_error = []
        self.grid_hd_reg_error = []
        self.grid_hd_update_filter = []
        self.grid_hd_cause_grad = []
        for i in range(len(layers['grid_hd'])):
            if i == 0:
                self.grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_minibatch[-1]), self.grid_hd_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_hd), self.grid_hd_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.grid_hd_bu_error += [reg_causes['grid_hd'][i] * (self.grid_hd_causes[i] ** 2)]
                if len(layers['grid_hd']) > 1:
                    raise NotImplementedError
                else:
                    self.grid_hd_cause_grad += [
                        tf.gradients([self.grid_hd_bu_error[i], self.grid_hd_bu_error[i+1], self.grid_hd_reg_error[i]],
                                               self.grid_hd_minibatch[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['grid_hd'][i] * (self.grid_hd_filters[i] ** 2)
                grid_hd_filter_grad = tf.gradients([self.grid_hd_bu_error[i], reg_error], self.grid_hd_filters[i])[0]
                self.grid_hd_update_filter += [
                    tf.assign_sub(self.grid_hd_filters[i], lr_filters['grid_hd'][i] * grid_hd_filter_grad)]
                reg_error = reg_filters['grid_hd'][i+1] * (self.grid_hd_filters[i+1] ** 2)
                grid_hd_filter_grad = tf.gradients([self.grid_hd_bu_error[i+1], reg_error], self.grid_hd_filters[i+1])[0]
                self.grid_hd_update_filter += [
                    tf.assign_sub(self.grid_hd_filters[i+1], lr_filters['grid_hd'][i+1] * grid_hd_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for computing gradients for vis_hd causes
        self.vis_hd_grid_hd_bu_error = []
        self.vis_hd_grid_hd_reg_error = []
        self.vis_hd_grid_hd_update_filter = []
        self.vis_hd_grid_hd_cause_grad = []
        for i in range(len(layers['(vis_hd)_(grid_hd)'])):
            if i == 0:
                self.vis_hd_grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_hd_minibatch[-1]), self.vis_hd_grid_hd_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_hd_grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_hd_minibatch[-1]), self.vis_hd_grid_hd_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.vis_hd_grid_hd_reg_error += [reg_causes['(vis_hd)_(grid_hd)'][i] * (self.vis_hd_grid_hd_causes[i] ** 2)]
                # self.vis_hd_reg_error += [tf.keras.regularizers.l2(reg_causes['vis_hd'][i])(self.vis_hd_minibatch[i])]
                if len(layers['(vis_hd)_(grid_hd)']) > 1:
                    raise NotImplementedError
                else:
                    self.vis_hd_grid_hd_cause_grad += [
                        tf.gradients([self.vis_hd_grid_hd_bu_error[i], self.vis_hd_grid_hd_bu_error[i+1], self.vis_hd_grid_hd_reg_error[i]],
                                               self.vis_hd_grid_hd_minibatch[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['(vis_hd)_(grid_hd)'][i] * (self.vis_hd_grid_hd_filters[i] ** 2)
                vis_hd_grid_hd_filter_grad = tf.gradients([self.vis_hd_grid_hd_bu_error[i], reg_error], self.vis_hd_grid_hd_filters[i])[0]
                self.vis_hd_grid_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_grid_hd_filters[i], lr_filters['(vis_hd)_(grid_hd)'][i] * vis_hd_grid_hd_filter_grad)]
                reg_error = reg_filters['(vis_hd)_(grid_hd)'][i+1] * (self.vis_hd_grid_hd_filters[i+1] ** 2)
                vis_hd_grid_hd_filter_grad = tf.gradients([self.vis_hd_grid_hd_bu_error[i+1], reg_error], self.vis_hd_grid_hd_filters[i+1])[0]
                self.vis_hd_grid_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_grid_hd_filters[i+1],  lr_filters['(vis_hd)_(grid_hd)'][i+1] * vis_hd_grid_hd_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for updating causes
        self.vis_update_cause = []
        self.hd_update_cause = []
        self.grid_update_cause = []
        self.vis_hd_update_cause = []
        self.grid_hd_update_cause = []
        self.vis_hd_grid_hd_update_cause = []
        with tf.control_dependencies(self.vis_cause_grad + self.grid_cause_grad + self.vis_hd_cause_grad + self.grid_hd_cause_grad + self.vis_hd_grid_hd_cause_grad):
            # vis modality
            for i in range(len(layers['vis'])):
                self.vis_update_cause += [tf.scatter_sub(self.vis_causes[i], indices=current_batch,
                                                                  updates=(lr_causes['vis'][i] * self.vis_cause_grad[i]))]

            # hd modality
            # for i in range(len(layers['hd'])):
            #     self.hd_update_cause += [tf.scatter_sub(self.hd_causes[i], indices=current_batch,
            #                                                       updates=(lr_causes['hd'][i] * self.hd_cause_grad[i]))]

            # grid modality
            for i in range(len(layers['grid'])):
                self.grid_update_cause += [tf.scatter_sub(self.grid_causes[i], indices=current_batch,
                                                                  updates=(lr_causes['grid'][i] * self.grid_cause_grad[i]))]

            # vis_hd modality
            for i in range(len(layers['vis_hd'])):
                self.vis_hd_update_cause += [tf.scatter_sub(self.vis_hd_causes[i], indices=current_batch,
                                                                   updates=(lr_causes['vis_hd'][i] * self.vis_hd_cause_grad[i]))]

            # grid_hd modality
            for i in range(len(layers['grid_hd'])):
                self.grid_hd_update_cause += [tf.scatter_sub(self.grid_hd_causes[i], indices=current_batch,
                                                                   updates=(lr_causes['grid_hd'][i] * self.grid_hd_cause_grad[i]))]

            # vis_hd_grid_hd modality
            for i in range(len(layers['(vis_hd)_(grid_hd)'])):
                self.vis_hd_grid_update_cause += [tf.scatter_sub(self.vis_hd_grid_hd_causes[i], indices=current_batch,
                                                                  updates=(lr_causes['(vis_hd)_(grid_hd)'][i] * self.vis_hd_grid_hd_causes[i]))]


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

            # hd_epoch_loss = np.vstack(
            #     [np.load('%s/hd_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['hd'])))])
            vis_hd_epoch_loss = np.vstack(
                [np.load('%s/vis_hd_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['vis_hd']) + 1))])

            vis_avg_activity = np.vstack(
                [np.load('%s/vis_avg_activity.npy' % load_path), np.zeros((n_epoch, len(layers['vis'])))])
            # hd_avg_activity = np.vstack(
            #     [np.load('%s/hd_avg_activity.npy' % load_path), np.zeros((n_epoch, len(layers['hd'])))])
            vis_hd_avg_activity = np.vstack(
                [np.load('%s/vis_hd_avg_activity.npy' % load_path), np.zeros((n_epoch, len(layers['vis_hd'])))])

            grid_epoch_loss = np.vstack(
                [np.load('%s/grid_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['grid']) + 1))])

            grid_avg_activity = np.vstack(
                [np.load('%s/grid_avg_activity.npy' % load_path), np.zeros((n_epoch, len(layers['grid'])))])

            grid_hd_epoch_loss = np.vstack(
                [np.load('%s/grid_hd_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['grid_hd']) + 1))])

            grid_hd_activity = np.vstack(
                [np.load('%s/grid_hd_activity.npy' % load_path), np.zeros((n_epoch, len(layers['grid_hd'])))])

            vis_hd_grid_hd_epoch_loss = np.vstack(
                [np.load('%s/vis_hd_grid_hd_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(layers['vis_hd_grid_hd']) + 1))])

            vis_hd_grid_hd_activity = np.vstack(
                [np.load('%s/vis_hd_grid_hd_activity.npy' % load_path), np.zeros((n_epoch, len(layers['vis_hd_grid_hd'])))])

        else:
            vis_epoch_loss = np.zeros((n_epoch, len(layers['vis'])))
            #hd_epoch_loss = np.zeros((n_epoch, len(layers['hd'])))
            vis_hd_epoch_loss = np.zeros((n_epoch, len(layers['vis_hd']) + 1))
            grid_epoch_loss = np.zeros((n_epoch, len(layers['grid'])))
            grid_hd_epoch_loss = np.zeros((n_epoch, len(layers['grid_hd']) + 1))
            vis_hd_grid_hd_epoch_loss = np.zeros((n_epoch, len(layers['(vis_hd)_(grid_hd)']) + 1))

            vis_avg_activity = np.zeros((n_epoch, len(layers['vis'])))
            #hd_avg_activity = np.zeros((n_epoch, len(layers['hd'])))
            vis_hd_avg_activity = np.zeros((n_epoch, len(layers['vis_hd']) + 1))
            grid_avg_activity = np.zeros((n_epoch, len(layers['grid'])))
            grid_hd_avg_activity = np.zeros((n_epoch, len(layers['grid_hd']) + 1))
            vis_hd_grid_hd_avg_activity = np.zeros((n_epoch, len(layers['(vis_hd)_(grid_hd)']) + 1))

        for i in range(n_epoch):
            current_epoch = completed_epoch + i

            n_batch = visual_data.shape[0] // minibatch_sz
            for j in range(n_batch):
                visual_batch = visual_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]
                head_direction_batch = head_direction_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]
                grid_batch = grid_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]

                # update causes
                for k in range(cause_epoch):
                    vis_cause, vis_hd_cause, grid_cause, grid_hd_cause, vis_hd_grid_hd_cause, vis_grad, vis_hd_grad, grid_grad, grid_hd_grad, vis_hd_grid_hd_grad, vis_hd_reg_error, grid_hd_reg_error, vis_hd_grid_hd_reg_error = sess.run(
                        [net.vis_update_cause, net.vis_hd_update_cause, net.grid_update_cause, net.grid_hd_update_cause, net.vis_hd_grid_hd_update_cause,
                        net.vis_cause_grad, net.vis_hd_cause_grad, net.grid_cause_grad, net.grid_hd_cause_grad, net.vis_hd_grid_hd_cause_grad, 
                        net.vis_hd_reg_error, net.grid_hd_reg_error, net.vis_hd_grid_reg_error],
                        feed_dict={net.x_vis: visual_batch, net.x_hd: head_direction_batch, net.x_grid: grid_batch, net.batch: j})

                # (optional) save reconstructions to diagnose training issues
                vis_reconstruction, vis_hd_reconstruction, grid_reconstruction, grid_hd_reconstruction, vis_hd_grid_reconstruction = sess.run([net.vis_predictions[0], net.vis_hd_predictions[1], net.grid_predictions[0], net.grid_hd_predictions[1], net.vis_hd_grid_predictions[0]],
                                                                feed_dict={net.x_vis: visual_batch, net.x_hd: head_direction_batch, net.x_grid: grid_batch, net.batch: j})

                if save_vis_causes:

                    np.save(cause_save_path + '/vis/epoch{}_batch{}_cause_0'.format(i, j), vis_cause[0])
                    np.save(cause_save_path + '/vis/epoch{}_batch{}_cause_1'.format(i, j), vis_cause[1])

                if save_vis_hd_causes:

                    np.save(cause_save_path + '/vis_hd/epoch{}_batch{}'.format(i, j), vis_hd_cause[0])

                if save_grid_causes:

                    np.save(cause_save_path + '/grid/epoch{}_batch{}_cause_0'.format(i, j), grid_cause[0])

                if save_grid_hd_causes:

                    np.save(cause_save_path + '/grid_hd/epoch{}_batch{}_cause_0'.format(i, j), grid_hd_cause[0])

                if save_vis_hd_grid_hd_causes:

                    np.save(cause_save_path + '/vis_hd_grid_hd/epoch{}_batch{}'.format(i, j), vis_hd_grid_hd_cause[0])

                if save_vis_reconstructions:

                    np.save(reconstruction_save_path + '/vis/epoch{}_batch{}_reconstruction.npy'.format(i, j), vis_reconstruction)

                if save_vis_hd_reconstructions:

                    np.save(reconstruction_save_path + '/vis_hd/epoch{}_batch{}_reconstruction.npy'.format(i, j), vis_hd_reconstruction)

                if save_grid_reconstructions:

                    np.save(reconstruction_save_path + '/grid/epoch{}_batch{}_reconstruction.npy'.format(i, j), grid_reconstruction)

                if save_vis_hd_grid_hd_reconstructions:

                    np.save(reconstruction_save_path + '/vis_hd_grid/epoch{}_batch{}_reconstruction.npy'.format(i, j), vis_hd_grid_reconstruction)

                # update weights
                _, _, _, _, vis_error, vis_hd_error, grid_error, grid_hd_error, vis_hd_grid_hd_error, vis_filter, vis_hd_filter, grid_filter, grid_hd_filter, vis_hd_grid_hd_filter = sess.run(
                    [net.vis_update_filter, net.vis_hd_update_filter, net.grid_update_filter, net.grid_hd_update_filter, net.vis_hd_grid_hd_update_filter,
                     net.vis_bu_error, net.vis_hd_bu_error, net.grid_bu_error, net.grid_hd_bu_error, net.vis_hd_grid_hd_bu_error,
                     net.vis_filters, net.vis_hd_filters, net.grid_filters, net.grid_hd_filters, net.vis_hd_grid_hd_filters],
                    feed_dict={net.x_vis: visual_batch, net.x_hd: head_direction_batch, net.x_grid: grid_batch, net.batch: j})

                # record maximum reconstruction error on the entire data
                vis_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > vis_epoch_loss[current_epoch, l]
                                                   else vis_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(vis_error)]
                #hd_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                #                                   if np.max(np.mean(item, axis=1)) > hd_epoch_loss[current_epoch, l]
                #                                   else hd_epoch_loss[current_epoch, l]
                #                                   for l, item in enumerate(hd_error)]
                vis_hd_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                    if np.max(np.mean(item, axis=1)) > vis_hd_epoch_loss[current_epoch, l]
                                                    else vis_hd_epoch_loss[current_epoch, l]
                                                    for l, item in enumerate(vis_hd_error)]
                grid_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > grid_epoch_loss[current_epoch, l]
                                                   else grid_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(grid_error)]
                grid_hd_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > grid_hd_epoch_loss[current_epoch, l]
                                                   else grid_hd_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(grid_hd_error)]
                vis_hd_grid_hd_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                    if np.max(np.mean(item, axis=1)) > vis_hd_grid_hd_epoch_loss[current_epoch, l]
                                                    else vis_hd_grid_hd_epoch_loss[current_epoch, l]
                                                    for l, item in enumerate(vis_hd_grid_hd_error)]

            # track average activity in inferred causes
            vis_avg_activity[current_epoch, :] = [np.mean(item) for item in vis_cause]
            #hd_avg_activity[current_epoch, :] = [np.mean(item) for item in hd_cause]
            vis_hd_avg_activity[current_epoch, :] = [np.mean(item) for item in vis_hd_cause]
            grid_avg_activity[current_epoch, :] = [np.mean(item) for item in grid_cause]
            grid_hd_avg_activity[current_epoch, :] = [np.mean(item) for item in grid_hd_cause]
            vis_hd_grid_hd_avg_activity[current_epoch, :] = [np.mean(item) for item in vis_hd_grid_hd_cause]

            print('-------- Epoch %d/%d --------\nVis Loss:%s Vis Mean Cause:%s\nGrid:%s Grid Mean Cause:%s\nVis_HD:%s Vis_HD Mean Cause:%s\nGrid_HD:%s Grid_HD Mean Cause:%s\nVis_HD_Grid_HD:%s Vis_HD_Grid_HD Mean Cause:%s' % (
                i, 
                n_epoch, 
                ', '.join(['%.8f' % elem for elem in vis_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in grid_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in grid_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_hd_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_hd_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in grid_hd_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in grid_hd_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_hd_grid_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in vis_hd_grid_avg_activity[current_epoch, :]])))

        # create the save path if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save model and stats
        saver.save(sess, '%s/main.ckpt' % save_path)
        np.save('%s/vis_epoch_loss.npy' % save_path, vis_epoch_loss)
        np.save('%s/grid_epoch_loss.npy' % save_path, grid_epoch_loss)
        np.save('%s/vis_hd_epoch_loss.npy' % save_path, vis_hd_epoch_loss)
        np.save('%s/grid_hd_epoch_loss.npy' % save_path, grid_hd_epoch_loss)
        np.save('%s/vis_hd_grid_hd_epoch_loss.npy' % save_path, vis_hd_grid_epoch_loss)
        np.save('%s/vis_avg_activity.npy' % save_path, vis_avg_activity)
        np.save('%s/grid_avg_activity.npy' % save_path, grid_avg_activity)
        np.save('%s/vis_hd_avg_activity.npy' % save_path, vis_hd_avg_activity)
        np.save('%s/grid_hd_avg_activity.npy' % save_path, grid_avg_activity)
        np.save('%s/vis_hd_grid_hd_avg_activity.npy' % save_path, vis_hd_grid_avg_activity)


if __name__ == '__main__':
    starttime = time.time()
    train()
    endtime = time.time()

    print ('Time taken: %f' % ((endtime - starttime) / 3600))

model_path = 'C:/Users/Tom/Downloads/HBP/model_checkpoints/visual_head_direction_grid_cells/stacked/'#trial1/'

test_sample_start = sample_end

test_sample_end = 5000

num_test_samps = test_sample_end - test_sample_start

error_criterion = np.array([1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3])

max_iter = 500

class InferenceNetwork:
    def __init__(self, inp_shape, layers, cause_init, reg_causes, lr_causes, available_modality='both'):

        # create placeholders
        self.x_vis = tf.placeholder(tf.float32, shape=[1, inp_shape['vis']])
        self.x_hd = tf.placeholder(tf.float32, shape=[1, inp_shape['hd']])
        self.x_grid = tf.placeholder(tf.float32, shape=[1, inp_shape['grid']])
        #self.batch = tf.placeholder(tf.int32, shape=[])

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

        # create filters and cause for head direction data (current taken care of by vis_hd, since there are no hidden layers for head direction at present)
        #self.hd_filters = []
        #self.hd_causes = []
        #for i in range(len(layers['hd'])):
        #    filter_name = 'hd_filter_%d' % i
        #    cause_name = 'hd_cause_%d' % i

        #    if i == 0:
        #        self.hd_filters += [tf.get_variable(filter_name, shape=[layers['hd'][i], inp_shape['hd']])]
        #    else:
        #        self.hd_filters += [tf.get_variable(filter_name, shape=[layers['hd'][i], layers['hd'][i-1]])]

        #    init = tf.constant_initializer(cause_init['hd'][i])
        #    self.hd_causes += [tf.get_variable(cause_name, shape=[n_sample, layers['hd'][i]], initializer=init)]

        # create filters and cause for grid cells
        self.grid_filters = []
        self.grid_causes = []
        for i in range(len(layers['grid'])):
            filter_name = 'grid_filter_%d' % i
            cause_name = 'grid_cause_%d' % i

            if i == 0:
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], inp_shape['grid']])]
            else:
                self.grid_filters += [tf.get_variable(filter_name, shape=[layers['grid'][i], layers['grid'][i-1]])]

            init = tf.constant_initializer(cause_init['grid'][i])
            self.grid_causes += [tf.get_variable(cause_name, shape=[1, layers['grid'][i]], initializer=init)]

        # create filters and cause for visual and head direction latent space
        self.vis_hd_filters = []
        self.vis_hd_causes = []
        for i in range(len(layers['vis_hd'])):
            if i == 0:
                # add filters for vis
                filter_name = 'vis_hd->vis_filter'
                self.vis_hd_filters += [tf.get_variable(filter_name, shape=[layers['vis_hd'][i],
                                                                                   layers['vis'][-1]])]
                # add filters for hd
                filter_name = 'vis_hd->hd_filter'
                self.vis_hd_filters += [tf.get_variable(filter_name, shape=[layers['vis_hd'][i],
                                                                                   inp_shape['hd']])]
            else:
                filter_name = 'vis_hd_filter_%d' % i
                self.vis_hd_filters += [tf.get_variable(filter_name, shape=[layers['vis_hd'][i],
                                                                                   layers['vis_hd'][i - 1]])]

            cause_name = 'vis_hd_cause_%d' % i
            init = tf.constant_initializer(cause_init['vis_hd'][i])
            self.vis_hd_causes += [tf.get_variable(cause_name, shape=[1, layers['vis_hd'][i]], initializer=init)]

        # create filters and cause for visual and head direction latent space
        self.grid_hd_filters = []
        self.grid_hd_causes = []
        for i in range(len(layers['grid_hd'])):
            if i == 0:
                # add filters for vis
                filter_name = 'grid_hd->grid_filter'
                self.grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['grid_hd'][i],
                                                                                   layers['grid'][-1]])]
                # add filters for hd
                filter_name = 'grid_hd->hd_filter'
                self.grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['grid_hd'][i],
                                                                                   inp_shape['hd']])]
            else:
                filter_name = 'grid_hd_filter_%d' % i
                self.grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['grid_hd'][i],
                                                                                   layers['grid_hd'][i - 1]])]

            cause_name = 'grid_hd_cause_%d' % i
            init = tf.constant_initializer(cause_init['grid_hd'][i])
            self.grid_hd_causes += [tf.get_variable(cause_name, shape=[1, layers['grid_hd'][i]], initializer=init)]

        # create filters and cause for 2nd tier latent space (visual+head_direction, with grid field)
        self.vis_hd_grid_hd_filters = []
        self.vis_hd_grid_hd_causes = []
        for i in range(len(layers['(vis_hd)_(grid_hd)'])):
            if i == 0:
                # add filters for vis
                filter_name = '(vis_hd)_(grid_hd)->grid_filter'
                self.vis_hd_grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['(vis_hd)_(grid_hd)'][i],
                                                                                   layers['grid_hd'][-1]])]
                # add filters for hd
                filter_name = '(vis_hd)_(grid_hd)->vis_hd_filter'
                self.vis_hd_grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['(vis_hd)_(grid_hd)'][i],
                                                                                   layers['vis_hd'][-1]])]
            else:
                filter_name = '(vis_hd)_(grid_hd)_filter_%d' % i
                self.vis_hd_grid_hd_filters += [tf.get_variable(filter_name, shape=[layers['(vis_hd)_(grid_hd)'][i],
                                                                                   layers['(vis_hd)_(grid_hd)'][i - 1]])]

            cause_name = '(vis_hd)_(grid_hd)_cause_%d' % i
            init = tf.constant_initializer(cause_init['(vis_hd)_(grid_hd)'][i])
            self.vis_hd_grid_hd_causes += [tf.get_variable(cause_name, shape=[1, layers['(vis_hd)_(grid_hd)'][i]], initializer=init)]

        # compute predictions
        #current_batch = tf.range(self.batch * minibatch_sz, (self.batch + 1) * minibatch_sz)
        # vis predictions
        self.vis_predictions = []
        for i in range(len(layers['vis'])):
            self.vis_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_causes[i], self.vis_filters[i]))]

        # hd predictions
        # self.hd_minibatch = []
        # self.hd_predictions = []
        # for i in range(len(layers['hd'])):
        #     self.hd_minibatch += [tf.gather(self.hd_causes[i], indices=current_batch, axis=0)]
        #     self.hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.hd_minibatch[i], self.hd_filters[i]))]

        # grid predictions
        self.grid_predictions = []
        for i in range(len(layers['grid'])):
            self.grid_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_causes[i], self.grid_filters[i]))]

        # vis_hd predictions
        self.vis_hd_predictions = []
        for i in range(len(layers['vis_hd'])):
            if i == 0:
                self.vis_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_causes[i], self.vis_hd_filters[i]))]  # vis prediction
                self.vis_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_causes[i], self.vis_hd_filters[i+1]))]  # hd prediction
            else:
                self.vis_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_causes[i], self.vis_hd_filters[i+1]))]

        # grid_hd predictions
        self.grid_hd_predictions = []
        for i in range(len(layers['grid_hd'])):
            if i == 0:
                self.grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_hd_causes[i], self.grid_hd_filters[i]))]  # grid prediction
                self.grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_hd_causes[i], self.grid_hd_filters[i+1]))]  # hd prediction
            else:
                self.grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.grid_hd_causes[i], self.grid_hd_filters[i+1]))]

        # vis_hd_grid_hd predictions
        self.vis_hd_grid_hd_predictions = []
        for i in range(len(layers['(vis_hd)_(grid_hd)'])):
            if i == 0:
                self.vis_hd_grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_grid_hd_causes[i], self.vis_hd_grid_hd_filters[i]))]  # grid_hd prediction
                self.vis_hd_grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_grid_hd_causes[i], self.vis_hd_grid_hd_filters[i+1]))]  # vis_hd prediction
            else:
                self.vis_hd_grid_hd_predictions += [tf.nn.leaky_relu(tf.matmul(self.vis_hd_grid_hd_causes[i], self.vis_hd_grid_hd_filters[i+1]))]

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
                    tf.stop_gradient(self.vis_hd_predictions[0]), self.vis_causes[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['vis'][i] * (self.vis_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_causes[i])
            self.vis_cause_grad += [tf.gradients([self.vis_bu_error[i], td_error, reg_error],
                                                          self.vis_causes[i])[0]]

            # ops for updating weights
            reg_error = reg_filters['vis'][i] * (self.vis_filters[i] ** 2)
            vis_filter_grad = tf.gradients([self.vis_bu_error[i], reg_error], self.vis_filters[i])[0]
            self.vis_update_filter += [
                tf.assign_sub(self.vis_filters[i], lr_filters['vis'][i] * vis_filter_grad)]

        # add ops for computing gradients for hd causes and for updating weights
        # self.hd_bu_error = []
        # self.hd_update_filter = []
        # self.hd_cause_grad = []
        # for i in range(len(layers['hd'])):
        #     if i == 0:
        #         self.hd_bu_error += [tf.losses.mean_squared_error(self.x_hd, self.hd_predictions[i],
        #                                                                     reduction=tf.losses.Reduction.NONE)]
        #     else:
        #         self.hd_bu_error += [tf.losses.mean_squared_error(
        #             tf.stop_gradient(self.hd_causes[i - 1]), self.hd_predictions[i],
        #             reduction=tf.losses.Reduction.NONE)]

        #     # compute top-down prediction error
        #     if len(layers['hd']) > (i + 1):
        #         # there are more layers in this modality
        #         td_error = tf.losses.mean_squared_error(
        #             tf.stop_gradient(self.hd_predictions[i+1]), self.hd_causes[i],
        #                 reduction=tf.losses.Reduction.NONE)
        #     else:
        #         # this is the only layer in this modality
        #         td_error = tf.losses.mean_squared_error(
        #             tf.stop_gradient(self.vis_hd_predictions[1]), self.hd_causes[i],
        #                 reduction=tf.losses.Reduction.NONE)

        #     #reg_error = reg_causes['hd'][i] * (self.hd_causes[i] ** 2)
        #     # reg_error = tf.keras.regularizers.l2(reg_causes['hd'][i])(self.hd_causes[i])
        #     reg_error = reg_causes['vis'][i] * ((self.vis_causes[i] - circular_mean(self.vis_causes[i], gaussian = False)) ** 2)
        #     self.hd_cause_grad += [
        #         tf.gradients([self.hd_bu_error[i], td_error, reg_error], self.hd_causes[i])[0]]

        #     # add ops for updating weights
        #     reg_error = reg_filters['hd'][i] * (self.hd_filters[i] ** 2)
        #     hd_filter_grad = tf.gradients([self.hd_bu_error[i], reg_error], self.hd_filters[i])[0]
        #     self.vis_update_filter += [
        #         tf.assign_sub(self.hd_filters[i], lr_hd_filters[i] * hd_filter_grad)]
        #     #else:
        #         #raise NotImplementedError

        # add ops for computing gradients for vis causes and for updating weights
        self.grid_bu_error = []
        self.grid_update_filter = []
        self.grid_cause_grad = []
        for i in range(len(layers['grid'])):
            if i == 0:
                self.grid_bu_error += [tf.losses.mean_squared_error(self.x_grid, self.grid_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.grid_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_causes[i - 1]), self.grid_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(layers['grid']) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_predictions[i+1]), self.grid_causes[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_hd_grid_predictions[0]), self.grid_causes[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_causes['grid'][i] * (self.grid_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_causes['vis'][i])(self.vis_causes[i])
            self.grid_cause_grad += [tf.gradients([self.grid_bu_error[i], td_error, reg_error],
                                                          self.grid_causes[i])[0]]

            # ops for updating weights
            reg_error = reg_filters['grid'][i] * (self.grid_filters[i] ** 2)
            grid_filter_grad = tf.gradients([self.grid_bu_error[i], reg_error], self.grid_filters[i])[0]
            self.grid_update_filter += [
                tf.assign_sub(self.grid_filters[i], lr_filters['grid'][i] * grid_filter_grad)]

        # add ops for computing gradients for vis_hd causes
        self.vis_hd_bu_error = []
        self.vis_hd_reg_error = []
        self.vis_hd_update_filter = []
        self.vis_hd_cause_grad = []
        for i in range(len(layers['vis_hd'])):
            if i == 0:
                self.vis_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_causes[-1]), self.vis_hd_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_hd), self.vis_hd_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.vis_hd_reg_error += [reg_causes['vis_hd'][i] * (self.vis_hd_causes[i] ** 2)]
                # self.vis_hd_reg_error += [tf.keras.regularizers.l2(reg_causes['vis_hd'][i])(self.vis_hd_causes[i])]
                if len(layers['vis_hd']) > 1:
                    raise NotImplementedError
                else:
                    self.vis_hd_cause_grad += [
                        tf.gradients([self.vis_hd_bu_error[i], self.vis_hd_bu_error[i+1], self.vis_hd_reg_error[i]],
                                               self.vis_hd_causes[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['vis_hd'][i] * (self.vis_hd_filters[i] ** 2)
                vis_hd_filter_grad = tf.gradients([self.vis_hd_bu_error[i], reg_error], self.vis_hd_filters[i])[0]
                self.vis_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_filters[i], lr_filters['vis_hd'][i] * vis_hd_filter_grad)]
                reg_error = reg_filters['vis_hd'][i+1] * (self.vis_hd_filters[i+1] ** 2)
                vis_hd_filter_grad = tf.gradients([self.vis_hd_bu_error[i+1], reg_error], self.vis_hd_filters[i+1])[0]
                self.vis_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_filters[i+1], lr_filters['vis_hd'][i+1] * vis_hd_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for computing gradients for vis_hd causes
        self.grid_hd_bu_error = []
        self.grid_hd_reg_error = []
        self.grid_hd_update_filter = []
        self.grid_hd_cause_grad = []
        for i in range(len(layers['grid_hd'])):
            if i == 0:
                self.grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_causes[-1]), self.grid_hd_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_hd), self.grid_hd_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.grid_hd_reg_error += [reg_causes['grid_hd'][i] * (self.grid_hd_causes[i] ** 2)]
                if len(layers['grid_hd']) > 1:
                    raise NotImplementedError
                else:
                    self.grid_hd_cause_grad += [
                        tf.gradients([self.grid_hd_bu_error[i], self.grid_hd_bu_error[i+1], self.grid_hd_reg_error[i]],
                                               self.grid_hd_causes[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['grid_hd'][i] * (self.grid_hd_filters[i] ** 2)
                grid_hd_filter_grad = tf.gradients([self.grid_hd_bu_error[i], reg_error], self.grid_hd_filters[i])[0]
                self.grid_hd_update_filter += [
                    tf.assign_sub(self.grid_hd_filters[i], lr_filters['grid_hd'][i] * grid_hd_filter_grad)]
                reg_error = reg_filters['grid_hd'][i+1] * (self.grid_hd_filters[i+1] ** 2)
                grid_hd_filter_grad = tf.gradients([self.grid_hd_bu_error[i+1], reg_error], self.grid_hd_filters[i+1])[0]
                self.grid_hd_update_filter += [
                    tf.assign_sub(self.grid_hd_filters[i+1], lr_filters['grid_hd'][i+1] * grid_hd_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for computing gradients for vis_hd causes
        self.vis_hd_grid_hd_bu_error = []
        self.vis_hd_grid_hd_reg_error = []
        self.vis_hd_grid_hd_update_filter = []
        self.vis_hd_grid_hd_cause_grad = []
        for i in range(len(layers['(vis_hd)_(grid_hd)'])):
            if i == 0:
                self.vis_hd_grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.grid_hd_causes[-1]), self.vis_hd_grid_hd_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.vis_hd_grid_hd_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.vis_hd_causes[-1]), self.vis_hd_grid_hd_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.vis_hd_grid_hd_reg_error += [reg_causes['(vis_hd)_(grid_hd)'][i] * (self.vis_hd_grid_hd_causes[i] ** 2)]
                if len(layers['(vis_hd)_(grid_hd)']) > 1:
                    raise NotImplementedError
                else:
                    self.vis_hd_grid_hd_cause_grad += [
                        tf.gradients([self.vis_hd_grid_hd_bu_error[i], self.vis_hd_grid_hd_bu_error[i+1], self.vis_hd_grid_hd_reg_error[i]],
                                               self.vis_hd_grid_hd_causes[i])[0]]

                # add ops for updating weights
                reg_error = reg_filters['(vis_hd)_(grid_hd)'][i] * (self.vis_hd_grid_hd_filters[i] ** 2)
                vis_hd_grid_hd_filter_grad = tf.gradients([self.vis_hd_grid_hd_bu_error[i], reg_error], self.vis_hd_grid_hd_filters[i])[0]
                self.vis_hd_grid_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_grid_hd_filters[i], lr_filters['(vis_hd)_(grid_hd)'][i] * vis_hd_grid_hd_filter_grad)]
                reg_error = reg_filters['(vis_hd)_(grid_hd)'][i+1] * (self.vis_hd_grid_hd_filters[i+1] ** 2)
                vis_hd_grid_hd_filter_grad = tf.gradients([self.vis_hd_grid_hd_bu_error[i+1], reg_error], self.vis_hd_grid_hd_filters[i+1])[0]
                self.vis_hd_grid_hd_update_filter += [
                    tf.assign_sub(self.vis_hd_grid_hd_filters[i+1],  lr_filters['(vis_hd)_(grid_hd)'][i+1] * vis_hd_grid_hd_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for updating causes
        self.vis_update_cause = []
        self.hd_update_cause = []
        self.grid_update_cause = []
        self.grid_hd_update_cause = []
        self.vis_hd_update_cause = []
        self.vis_hd_grid_update_cause = []
        with tf.control_dependencies(self.vis_cause_grad + self.grid_cause_grad + self.vis_hd_cause_grad + self.grid_hd_cause_grad + self.vis_hd_grid_hd_cause_grad):
            # vis modality
            for i in range(len(layers['vis'])):
                self.vis_update_cause += [tf.assign_sub(self.vis_causes[i], (lr_causes['vis'][i] * self.vis_cause_grad[i]))]

            # hd modality
            # for i in range(len(layers['hd'])):
            #     self.hd_update_cause += [tf.scatter_sub(self.hd_causes[i], indices=current_batch,
            #                                                       updates=(lr_causes['hd'][i] * self.hd_cause_grad[i]))]

            # grid modality
            for i in range(len(layers['grid'])):
                self.grid_update_cause += [tf.assign_sub(self.grid_causes[i], (lr_causes['grid'][i] * self.grid_cause_grad[i]))]

            # vis_hd modality
            for i in range(len(layers['vis_hd'])):
                self.vis_hd_update_cause += [tf.assign_sub(self.vis_hd_causes[i], (lr_causes['vis_hd'][i] * self.vis_hd_cause_grad[i]))]

            # grid_hd modality
            for i in range(len(layers['grid_hd'])):
                self.grid_hd_update_cause += [tf.assign_sub(self.grid_hd_causes[i], (lr_causes['grid_hd'][i] * self.grid_hd_cause_grad[i]))]

            # vis_hd_grid_hd modality
            for i in range(len(layers['(vis_hd)_(grid_hd)'])):
                self.vis_hd_grid_hd_update_cause += [tf.assign_sub(self.vis_hd_grid_hd_causes[i], (lr_causes['vis_hd_grid_hd'][i] * self.vis_hd_grid_hd_cause_grad[i]))]

def init_network(model_path, available_modality='all'):
    tf.reset_default_graph()

    net = InferenceNetwork(inp_shape, layers, cause_init, reg_causes, lr_causes, available_modality)

    saver = tf.compat.v1.train.Saver(net.vis_filters + net.grid_filters + net.vis_hd_filters + net.grid_hd_filters + net.vis_hd_grid_hd_filters)
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, '%smain.ckpt' % model_path)

    return sess, net

def infer_repr(sess, net, max_iter=1, error_criterion=0.0001, visual_data=None, head_direction_data=None, verbose=False,
               available_modality='all'):
    iter = 1

    while True:
        # infer representations
        vis_cause, vis_hd_cause, grid_cause, vis_error, vis_hd_error, grid_error, vis_pred, vis_hd_pred, grid_pred, vis_filter, vis_hd_filter, grid_filter = sess.run(
                [net.vis_update_cause, net.vis_hd_update_cause, net.grid_update_cause,
                 net.vis_bu_error, net.vis_hd_bu_error, net.grid_bu_error,
                 net.vis_predictions, net.vis_hd_predictions, net.grid_predictions,
                 net.vis_filters, net.vis_hd_filters, net.grid_filters],
                 feed_dict={net.x_vis: visual_data, net.x_hd: head_direction_data, net.x_grid: grid_data})

        if 'visual' in available_modalities:
            vis_epoch_loss = [np.mean(item) for item in vis_error]
        else:
            vis_epoch_loss = [np.NINF for item in vis_error]

        if 'hd' in available_modalities:
            hd_epoch_loss = [vis_hd_error[1], grid_hd_error[1]]
        else:
            hd_epoch_loss = [np.NINF, np.NINF]

        if 'grid' in available_modalities:
            grid_epoch_loss = [np.mean(item) for item in grid_error]
        else:
            grid_epoch_loss = [np.NINF for item in grid_error]

        # if 'visual' in available_modalities and 'hd' in available_modalities:
        #     vis_hd_epoch_loss = [np.mean(item) for item in vis_hd_error]
        # else:
        #     vis_hd_epoch_loss = [np.NINF for item in vis_hd_error]

        # if 'grid' in available_modalities and 'hd' in available_modalities:
        #     grid_hd_epoch_loss = [np.mean(item) for item in grid_hd_error]
        # else:
        #     grid_hd_epoch_loss = [np.NINF for item in grid_hd_error]

        # if 'visual' in available_modalities and 'grid' in available_modalities and 'hd' in available_modalities:
        #     vis_hd_grid_hd_epoch_loss = [np.mean(item) for item in vis_hd_grid_hd_error]
        # else:
        #     vis_hd_grid_hd_epoch_loss = [np.NINF for item in vis_hd_grid_hd_error]


        if (np.all(np.array(vis_epoch_loss + hd_epoch_loss, grid_epoch_loss) < error_criterion)) or (iter >= max_iter):
            if verbose:
                print_str = ', '.join(['%.8f' % elem for elem in vis_epoch_loss + hd_epoch_loss + grid_epoch_loss])
                print ('(%d) %s' % (iter, print_str))
            break
        else:
            iter += 1

    # reconstruct the missing modality
    recon_head_direction = np.dot(vis_hd_cause[0], vis_hd_filter[1])
    #for l in range(len(hd_filter), 0, -1):
    #    recon_head_direction = np.dot(recon_head_direction, hd_filter[l - 1])
    recon_vis = np.dot(vis_hd_cause[0], vis_hd_filter[0])
    for l in range(len(vis_filter), 0, -1):
        recon_vis = np.dot(recon_vis, vis_filter[l - 1])

    return vis_hd_cause, recon_vis, recon_head_direction

def run_inference(sess, net, visual_data, head_direction_data, representations_save_path, available_modality):

    print("Modality: {}, Inferences will be saved to: {}".format(avail_modality, representations_save_path))

    print(visual_data.shape)
    print(head_direction_data.shape)
    print(grid_data.shape)

    assert visual_data.shape[0] == head_direction_data.shape[0]
    assert visual_data.shape[0] == grid_data.shape[0]

    recon_temp_vis = np.zeros([ visual_data.shape[0], 45, 80, 3 ])
    recon_temp_hd = np.zeros([ head_direction_data.shape[0] , 180 ])
    recon_temp_grid = np.zeros([ grid_data.shape[0] , 360 ])
    representations = np.zeros([head_direction_data.shape[0], 100])

    sess.run(tf.compat.v1.variables_initializer(net.vis_causes + net.hd_causes + net.grid_causes + net.vis_hd_causes + net.grid_hd_causes + net.vis_hd_grid_hd_causes))

    for j in range (visual_data.shape[0]):

        visual_input = visual_data[None, j]
        head_direction_input = head_direction_data[None, j]
        grid_input = head_direction_data[None, j]

        if 'visual' not in available_modalities:
            visual_input = np.zeros([1, 10800])

        if 'hd' not in available_modalities:
            head_direction_input = np.zeros([1, 180])

        if 'grid' not in available_modalities:
            grid_input = np.zeros([1, 360])

        reps, recon_vis, recon_hd, recon_grid = infer_repr(sess, net, max_iter, error_criterion, visual_input, head_direction_input, grid_input, True, available_modalities)
        representations[j, :] = reps[0]
        recon_temp_vis[j, :] = recon_vis.reshape(45,80,3) # reform into image
        recon_temp_hd[j, :] = recon_hd
        recon_temp_grid[j, :] = recon_grid

    print('Done!')

    np.save(representations_save_path + 'representations.npy', representations)
    sio.savemat(representations_save_path + 'reps.mat', {'reps':representations})

    np.save(representations_save_path + 'reconstructions_visual.npy', recon_temp_vis)
    sio.savemat(representations_save_path + 'recon_vis.mat', {'recon_vis':recon_temp_vis})

    np.save(representations_save_path + 'reconstructions_hd_cells.npy', recon_temp_hd)
    sio.savemat(representations_save_path + 'recon_hd.mat', {'recon_hd':recon_temp_hd})

    np.save(representations_save_path + 'reconstructions_grid_cells.npy', recon_temp_grid)
    sio.savemat(representations_save_path + 'recon_grid.mat', {'recon_grid':recon_temp_grid})

def generate_representations(shuffle):

    root_test_path = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_grid_cells_"
    root_representations_path = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_grid_cells_"

    #dataset = 1
    #if True:
    if False:
    #for dataset in (1,6):#,10,11,12):#13,14,15,16,17,18,19,20):#for dataset in (9,10,11,12,1,2,3,4,5,6,7,8):

        test_set = root_test_path + "{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_baseline/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}/both/".format(dataset)

        visual_data, head_direction_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)
        print(visual_data.shape)

        sess, net = init_network(model_path, available_modality = ['visual', 'hd', 'grid'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, head_direction_data, representations_save_path, available_modality = ['visual', 'hd', 'grid'])

    if True:
        dataset = "test_dataset"
    #if False:
    #for dataset in (1,6):#,10,11,12):#for dataset in (1,11):
    #for dataset in ("circling_proximal", "random_distal", "random_proximal", "cogarch"): #("rotating_distal", "rotating_proximal", "circling_distal", "circling_proximal", "random_distal", "random_proximal", "cogarch"):
    #for dataset in ("random_distal_2", "random_distal_3", "random_distal_4", "random_distal_5"):

        test_set = root_test_path + "{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_blind_blackout/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}/visual_only/".format(dataset)

        visual_data, head_direction_data, grid_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modality = ['visual'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, head_direction_data, representations_save_path, available_modality = ['visual'])

    #if True:
    if False:
    #for dataset in (1,6):#,10,11,12):#for dataset in (9,10,11,12,1,2,3,4,5,6,7,8):

        test_set = root_test_path + "{}".format(dataset)
        #reconstructions_save_path = "C:/Users/Tom/Downloads/HBP/results/fully_trained_reconstructions/matlab_physical_numb_blackout/testset{}/".format(dataset)
        representations_save_path = root_representations_path + "{}/head_direction_only/".format(dataset)

        visual_data, head_direction_data = load_npy_data(test_set, test_sample_start, test_sample_end, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modality = ['hd'])

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, head_direction_data, representations_save_path, available_modality = ['hd'])

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

body_head_direction = np.load(data_path + '/body_pose.npy')[:,0]

np.save(data_path + '/training_timestamps.npy', body_head_direction[sample_start:sample_end])

np.save(data_path + '/test_timestamps.npy', body_head_direction[test_sample_start:test_sample_end])