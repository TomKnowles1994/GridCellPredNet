import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
from math import prod, exp
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm
from matplotlib.animation import FuncAnimation

colourtest = False

def rings_to_som(data_path, sample_count = 100, map_dimensions = (9,9), topology = 'square', ring_scale = 30, ring_type = 'index', neighbourhood = 5, epochs = 100, learning_rate = 0.01, wraparound = False):

    input_data = np.array([np.load(data_path + 'ring_{}_gaussians_{}.npy'.format(i, ring_scale)) for i in (1,2,3)])[:,:sample_count]

    #print(input_data.shape)

    if ring_type == 'index' and len(input_data.shape) > 2:
    
        input_data = np.transpose(input_data, (1, 0, 2))

        input_data = np.argmax(input_data, axis = -1)

        input_data = input_data / ring_scale

    else:

        input_data = np.transpose(input_data, (1, 0, 2))

    if colourtest:

        input_generator = np.random.default_rng()

        input_data = input_generator.random(size = (sample_count, 3))

    print(input_data.shape)

    random_generator = np.random.default_rng()

    map_weights = random_generator.random(size = (*map_dimensions, *input_data.shape[1:]))

    # map_weight_means = random_generator.random((*map_dimensions, *input_data.shape[1:-1]))

    # print(map_weight_means.shape)

    # map_weights = norm.pdf(np.mgrid[0:map_dimensions[0], 0:map_dimensions[1], 0:input_data.shape[1], 0:input_data.shape[2]], loc = map_weight_means)

    #print(map_weights.shape)

    map_indices = np.dstack(np.indices(dimensions = map_dimensions))[:,:,::-1]

    #print(map_indices.shape)

    neighbourhood = neighbourhood

    learning_rate = learning_rate

    frames = []

    for epoch in range(epochs):

        theta = np.random.default_rng()

        theta_choice = theta.random()

        #if True:
        for item in input_data:

            if theta_choice > 0.75:

            #if True:
                fig, ax = plt.subplots(3,3)

                #item = input_data[0]

                difference =                            map_weights - item

                input_distance =                        np.linalg.norm(difference, ord = 2, axis = -1)

                winning_node_flat =                     np.argmin(input_distance.flatten())

                winning_node_x =                        winning_node_flat // map_dimensions[0]

                winning_node_y =                        winning_node_flat % map_dimensions[0]

                winning_node =                          (winning_node_x, winning_node_y)

                for artist in ax.flatten():

                    artist.scatter(*winning_node, c = 'red')

                weight_difference =                     np.abs(map_indices - winning_node)

                if wraparound:

                    weight_difference_left =            np.abs(map_indices - (winning_node + np.array(map_dimensions)))
                    weight_difference_right =           np.abs(map_indices - (winning_node - np.array(map_dimensions)))

                    weight_difference =                 np.min([weight_difference, weight_difference_left, weight_difference_right], axis = 0)

                winning_node_distance =                 np.linalg.norm(weight_difference, ord = 2, axis = -1)

                winning_node_distance_gaussian =        np.exp((-winning_node_distance ** 2) / (2 * neighbourhood ** 2))

                ax[0][0].imshow(((winning_node_distance - np.min(winning_node_distance)) / (np.max(winning_node_distance) - np.min(winning_node_distance))), vmin = 0, vmax = 1, origin = 'lower')
                ax[0][1].imshow(((winning_node_distance - np.min(winning_node_distance)) / (np.max(winning_node_distance) - np.min(winning_node_distance))), vmin = 0, vmax = 1, origin = 'lower')
                ax[0][2].imshow(((winning_node_distance - np.min(winning_node_distance)) / (np.max(winning_node_distance) - np.min(winning_node_distance))), vmin = 0, vmax = 1, origin = 'lower')

                gradients =                             learning_rate * difference * winning_node_distance_gaussian[..., None]

                ax[1][0].imshow(((gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients)))[...,0], vmin = 0, vmax = 1, origin = 'lower')
                ax[1][1].imshow(((gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients)))[...,1], vmin = 0, vmax = 1, origin = 'lower')
                ax[1][2].imshow(((gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients)))[...,2], vmin = 0, vmax = 1, origin = 'lower')

                for artist in ax.flatten():

                    gradients_sum =                     np.sum(gradients, axis = -1)

                    print(gradients_sum.shape)

                    gradients_flat =                    np.argmax(gradients_sum.flatten())

                    print(gradients_flat.shape)

                    gradients_x =                       gradients_flat // map_dimensions[0]

                    gradients_y =                       gradients_flat % map_dimensions[0]

                    gradient_max =                      (gradients_x, gradients_y)

                    artist.scatter(*gradient_max, c = 'orange', s = 40)

                print(gradient_max)

                gradients[winning_node_distance > neighbourhood] =  0

                ax[2][0].imshow(((gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients)))[...,0], vmin = 0, vmax = 1, origin = 'lower')
                ax[2][1].imshow(((gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients)))[...,1], vmin = 0, vmax = 1, origin = 'lower')
                ax[2][2].imshow(((gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients)))[...,2], vmin = 0, vmax = 1, origin = 'lower')

                map_weights = map_weights + gradients

                plt.show()

        #print(np.argmin(difference_summed))
        #print(difference_summed.shape)

        neighbourhood = neighbourhood * np.exp(-0.01)
        learning_rate = learning_rate * np.exp(-0.01)

        #print(neighbourhood)
        #print(learning_rate)

        # print(difference.shape)
        # print(winning_node)
        # print(xy_difference.shape)

        # fig, ax = plt.subplots(1,3)

        # ax[0].imshow(difference[..., 0])
        # ax[1].imshow(difference[..., 1])
        # ax[2].imshow(difference[..., 2])

        # plt.show()
        frames.append(map_weights)

        #print("Max map weight: {}".format(np.max(map_weights)))
        #print("Min map weight: {}".format(np.min(map_weights)))

    #print(euclidean_distance.shape)
    #print(euclidean_distance_sum.shape)
    #print(gaussian_difference)

    return frames

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/real_rat/'

frames = rings_to_som(data_path, sample_count = 100, map_dimensions = (10, 10), 
                      neighbourhood = 5, learning_rate = 0.01, topology = 'square', 
                      wraparound = False)

fig, ax = plt.subplots(1,4)

#print("Display frames shape: {}".format(frames[0][...,0].shape))

map1 = ax[0].imshow(frames[0][...,0])
map2 = ax[1].imshow(frames[0][...,1])
map3 = ax[2].imshow(frames[0][...,2])
map4 = ax[3].imshow(frames[0][...,:])

def animate_som_learning(i):

    #print(i)

    map1.set_data(frames[i][...,0])
    map2.set_data(frames[i][...,1])
    map3.set_data(frames[i][...,2])
    map4.set_data(frames[i][...,:])

    return map1, map2, map3, map4

#assert not np.array_equal(frames[0], frames[1])

#print("Frame 0: {}".format(frames[0]))
#print("Frame 1: {}".format(frames[1]))

ani = FuncAnimation(fig, animate_som_learning, frames = range(len(frames)), interval = 100, blit = True)

plt.show()