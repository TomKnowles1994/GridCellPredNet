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

show_gradients = False

def rings_to_som(data_path, sample_count = 100, map_dimensions = (10,10), topology = 'square', 
                 ring_scale = 30, ring_type = 'index', neighbourhood = 5, neighbourhood_decay = 1,
                 epochs = 100, learning_rate = 0.01, learning_rate_decay = 1, wraparound = False,
                 mode = 'training', weights = None):

    input_data = np.array([np.load(data_path + 'ring_{}_gaussians_{}.npy'.format(i, ring_scale)) for i in (1,2,3)])[:,:sample_count]

    if ring_type == 'index' and len(input_data.shape) > 2:
    
        input_data = np.transpose(input_data, (1, 0, 2))

        input_data = np.argmax(input_data, axis = -1)

        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

    else:

        input_data = np.transpose(input_data, (1, 0, 2))

    if colourtest:

        input_generator = np.random.default_rng()

        input_data = input_generator.random(size = (sample_count, 3))

    print(input_data.shape)

    if weights is None:

        random_generator = np.random.default_rng()

        map_weights = random_generator.random(size = (*map_dimensions, *input_data.shape[1:]))

    else:

        map_weights = weights

    map_indices = np.dstack(np.indices(dimensions = map_dimensions))[:,:,::-1]

    neighbourhood = neighbourhood

    learning_rate = learning_rate

    frames = []

    theta = np.random.default_rng()

    if mode == 'training':

        for epoch in range(epochs):
            
            input_data = theta.permutation(input_data, axis = 0)

            for item in input_data:

                #if theta.random() > 0.75:
                if True:

                    difference =                            map_weights - item

                    input_distance =                        np.linalg.norm(difference, ord = 2, axis = -1)

                    winning_node_flat =                     np.argmin(input_distance.flatten())

                    winning_node_x =                        winning_node_flat // map_dimensions[0]

                    winning_node_y =                        winning_node_flat % map_dimensions[0]

                    winning_node =                          (winning_node_x, winning_node_y)

                    weight_difference =                     np.abs(map_indices - winning_node)

                    if wraparound:

                        weight_difference_left =            np.abs(map_indices - (winning_node + np.array(map_dimensions)))
                        weight_difference_right =           np.abs(map_indices - (winning_node - np.array(map_dimensions)))

                        weight_difference =                 np.min([weight_difference, weight_difference_left, weight_difference_right], axis = 0)

                    winning_node_distance =                 np.linalg.norm(weight_difference, ord = 2, axis = -1)

                    winning_node_distance_gaussian =        np.exp((-winning_node_distance ** 2) / (neighbourhood ** 2))

                    gradients =                             learning_rate * difference * winning_node_distance_gaussian[..., None]

                    if show_gradients:

                        fig, ax = plt.subplots(4,3)

                        for artist in ax.flatten():

                            artist.scatter(*winning_node, c = 'red')

                        ax[0][0].imshow(((winning_node_distance - np.min(winning_node_distance)) / (np.max(winning_node_distance) - np.min(winning_node_distance))), vmin = 0, vmax = 1, origin = 'lower')
                        ax[0][1].imshow(((winning_node_distance - np.min(winning_node_distance)) / (np.max(winning_node_distance) - np.min(winning_node_distance))), vmin = 0, vmax = 1, origin = 'lower')
                        ax[0][2].imshow(((winning_node_distance - np.min(winning_node_distance)) / (np.max(winning_node_distance) - np.min(winning_node_distance))), vmin = 0, vmax = 1, origin = 'lower')

                        ax[0][0].set_ylabel("Distance")

                        ax[1][0].imshow(((winning_node_distance_gaussian - np.min(winning_node_distance_gaussian)) / (np.max(winning_node_distance_gaussian) - np.min(winning_node_distance_gaussian))), vmin = 0, vmax = 1, origin = 'lower')
                        ax[1][1].imshow(((winning_node_distance_gaussian - np.min(winning_node_distance_gaussian)) / (np.max(winning_node_distance_gaussian) - np.min(winning_node_distance_gaussian))), vmin = 0, vmax = 1, origin = 'lower')
                        ax[1][2].imshow(((winning_node_distance_gaussian - np.min(winning_node_distance_gaussian)) / (np.max(winning_node_distance_gaussian) - np.min(winning_node_distance_gaussian))), vmin = 0, vmax = 1, origin = 'lower')

                        ax[1][0].set_ylabel("Gaussian")

                        for i in range(gradients.shape[-1]):

                            ax[2][i].imshow(((gradients[..., i] - np.min(gradients[..., i])) / (np.max(gradients[..., i]) - np.min(gradients[..., i]))), vmin = 0, vmax = 1, origin = 'lower')
                            
                        ax[2][0].set_ylabel("\u0394W, raw")

                        for artist in ax.flatten():

                            gradients_sum =                     np.sum(gradients ** 2, axis = -1)

                            print(gradients_sum.shape)

                            gradients_flat =                    np.argmax(gradients_sum.flatten())

                            print(gradients_flat.shape)

                            gradients_x =                       gradients_flat // map_dimensions[0]

                            gradients_y =                       gradients_flat % map_dimensions[0]

                            gradient_max =                      (gradients_x, gradients_y)

                            artist.scatter(*gradient_max, c = 'orange', s = 40)

                        print(gradient_max)

                    gradients[winning_node_distance > neighbourhood] =  0

                    if show_gradients:
                        
                        for i in range(gradients.shape[-1]):

                            ax[3][i].imshow(((gradients[..., i] - np.min(gradients[..., i])) / (np.max(gradients[..., i]) - np.min(gradients[..., i]))), vmin = 0, vmax = 1, origin = 'lower')
                            
                        ax[3][0].set_ylabel("\u0394W, clipped")

                    map_weights = map_weights + gradients

                    for i in range(map_weights.shape[-1]):

                        map_weights[:,:,i] = (map_weights[:,:,i] - np.min(map_weights[:,:,i])) / (np.max(map_weights[:,:,i]) - np.min(map_weights[:,:,i]))

                    plt.show()

            if neighbourhood > 1:# and epoch > epochs // 10:

                neighbourhood = neighbourhood * np.exp(-neighbourhood_decay)

            if learning_rate > 0.000001:# and epoch > epochs // 10:

                learning_rate = learning_rate * np.exp(-learning_rate_decay)

            frames.append(map_weights)

        return frames

    elif mode == 'inference':

        inferences = np.zeros(shape = (input_data.shape))

        for i, item in enumerate(input_data):

            difference =                            map_weights - item

            input_distance =                        np.linalg.norm(difference, ord = 2, axis = -1)

            winning_node_flat =                     np.argmin(input_distance.flatten())

            winning_node_x =                        winning_node_flat // map_dimensions[0]

            winning_node_y =                        winning_node_flat % map_dimensions[0]

            winning_node =                          (winning_node_x, winning_node_y)

            inferences[i, :] =                      map_weights[winning_node[0], winning_node[1], :]

        return inferences

data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/real_rat/'

# frames = rings_to_som(  data_path, epochs = 5, sample_count = 3000, map_dimensions = (10, 10), 
#                         neighbourhood = 5, neighbourhood_decay = 0.1, 
#                         learning_rate = 0.1, learning_rate_decay = 0.1,
#                         topology = 'square', wraparound = True)

frames = rings_to_som(  data_path, epochs = 100, sample_count = 3000, map_dimensions = (10, 10), 
                        neighbourhood = 5, neighbourhood_decay = 0.001, 
                        learning_rate = 0.01, learning_rate_decay = 0.01,
                        topology = 'square', wraparound = True, mode = 'training')

fig, ax = plt.subplots(1,4)

map1 = ax[0].imshow(frames[0][...,0], cmap = 'Reds', vmin = 0, vmax = 1, origin = 'lower')
map2 = ax[1].imshow(frames[0][...,1], cmap = 'Greens', vmin = 0, vmax = 1, origin = 'lower')
map3 = ax[2].imshow(frames[0][...,2], cmap = 'Blues', vmin = 0, vmax = 1, origin = 'lower')
map4 = ax[3].imshow(frames[0][...,:], vmin = 0, vmax = 1, origin = 'lower')

if colourtest is False:

    ax[0].set_xlabel("Ring 1")
    ax[1].set_xlabel("Ring 2")
    ax[2].set_xlabel("Ring 3")
    ax[3].set_xlabel("State Space")

elif colourtest is True:

    ax[0].set_xlabel("Red")
    ax[1].set_xlabel("Green")
    ax[2].set_xlabel("Blue")
    ax[3].set_xlabel("Colour Space")

def animate_som_learning(i):

    print("Epoch {}/{}".format(i+1, len(frames)), end = '\r')

    map1.set_data(frames[i][...,0])
    map2.set_data(frames[i][...,1])
    map3.set_data(frames[i][...,2])
    map4.set_data(frames[i][...,:])

    return map1, map2, map3, map4

ani = FuncAnimation(fig, animate_som_learning, frames = range(len(frames)), interval = 100, blit = True, repeat = False)

plt.show()
