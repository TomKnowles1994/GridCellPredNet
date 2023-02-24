import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from math import prod, exp
from tensorflow.keras import backend as K

class SOM(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__({**kwargs, "dynamic": True})

        self.learning_rate = tf.Variable(1.)
        self.radius = tf.Variable(5.)

    def build(self, input_shape, map_shape = (9, 9)):

        print("Build input shape: {}".format(input_shape))

        self.map_shape = map_shape

        # Create manifold of 'weights'

        self.map_weights = self.add_weight( shape = (*self.map_shape, *input_shape[1:]), #prod(input_shape[1:])),
                                            initializer = 'random_normal',
                                            trainable = False)

        self.output_weights = None

        # Preallocate Tensorflow variables

        self.map_indices = tf.Variable(np.dstack(np.indices(dimensions = self.map_weights.shape[:2])))

        self.winning_node = tf.Variable([0, 0], dtype = tf.int32)

        super().build(input_shape)

    def call(self, inputs):

        print("Call input shape: {}".format(inputs.shape))
        print("Backend Tensor input shape: {}".format(K.shape(inputs)))
        print("Backend Tensor input shape: {}".format(K.int_shape(inputs)))

        # Batch size is not known at build time, therefore the output shape can only be defined on the first call

        if self.output_weights is None:

            #self.output_weights = tf.zeros(shape = (inputs.shape[0], *self.map_weights.shape), dtype = tf.float32)
            self.output_weights = tf.zeros(shape = (K.shape(inputs)[0], *self.map_weights.shape), dtype = tf.float32)

        # When called, reduce learning rate and radius

        self.learning_rate =            self.learning_rate * exp(-0.01)
        self.radius =                   self.radius * exp(-0.01)

        # Sequentially mutate graph to each batch item in turn
        # This effectively means that batching doesn't matter for this layer, but allows it to work with batch-aware layers

        for i, batch_item in enumerate(inputs):

            # Calculate the distance between every node in the map and the incoming data item

            print("Batch item: {}".format(batch_item))

            euclidean_distance =            tf.math.reduce_euclidean_norm([self.map_weights - batch_item], axis = 0)

            euclidean_distance_sum_rings =  tf.reduce_sum(euclidean_distance, axis = [2,3])

            # Find the best (closest) node

            winning_node_flat =             tf.math.argmin(tf.reshape(euclidean_distance, shape = (prod(euclidean_distance.shape), )))

            winning_node_x =                winning_node_flat // self.map_shape[0]

            winning_node_y =                winning_node_flat % self.map_shape[0]

            #winning_node =                  tf.Variable([winning_node_x, winning_node_y], dtype = tf.int32)

            self.winning_node               .assign([winning_node_x, winning_node_y])

            # Calculate distance in index space

            xy_distance =                   self.map_indices - self.winning_node

            xy_distance =                   tf.cast(xy_distance, tf.float32)

            # Use this to calculate the distance of every node from the winning node

            winning_node_distance =         tf.math.sqrt(tf.math.square(xy_distance[..., 0]) + tf.math.square(xy_distance[..., 1]))

            winning_node_distance =         tf.math.exp((-winning_node_distance ** 2) / (2 * self.radius ** 2))

            winning_node_distance =         tf.cast(winning_node_distance, dtype = tf.float32)

            # Apply gradient proportional to:-

            # - the current learning rate
            # - how far each node is from the winning node, in index space
            # - whether the node is within the radius of influence (Y/N)
            # - the difference between the node's value and that of the input

            gradients =                     self.learning_rate * winning_node_distance 
            
            gradients =                     gradients * euclidean_distance_sum_rings

            gradients =                     gradients[gradients < self.radius]

            self.map_weights = self.map_weights + gradients

            self.output_weights[i, ...] = self.map_weights

        return self.output_weights

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data_vis_hd_grid/'

ring_scale = 30

input_data = np.array([np.load(data_path + 'ring_{}_gaussians_{}.npy'.format(i, ring_scale)) for i in (1,2,3)])[:,:100]

input_data = np.transpose(input_data, (1, 0, 2)) # Take from (ring, row, gaussian) to (row, ring, gaussian)

output_data = np.load(data_path + 'filtered_body_poses.npy')[:100, 1:]
output_data[:, 2][output_data[:, 2] < 0] = output_data[:, 2][output_data[:, 2] < 0] + 2 * np.pi
output_data = (output_data - np.min(output_data, axis = 0)) / (np.max(output_data, axis = 0) - np.min(output_data, axis = 0))

input_layer =   keras.Input(shape = input_data.shape[1:], name = 'Input')
som =           SOM(name = 'SOM')(input_layer)
output_xy =     keras.layers.Dense(2, activation='relu', name = 'XY')(som)
output_theta =  keras.layers.Dense(1, activation='relu', name = 'Theta')(som)

model = keras.Model(inputs = input_layer, outputs = [output_xy, output_theta])

model.compile(
    optimizer = keras.optimizers.Adam(1e-7),
    loss = 'mse'
)

print(input_data.shape)
print(output_data.shape)

model.fit(x = input_data, y = output_data, validation_data = (input_data, output_data), batch_size = 5, epochs = 10, verbose = 2)