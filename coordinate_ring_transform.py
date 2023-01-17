import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace

data_filepath = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

poses = np.load(data_filepath + "filtered_body_poses.npy")[:,1:]

fig, ax = plt.subplots(1,1)

ax.set_xlim(-3.3, 3.3)
ax.set_ylim(-3.3, 3.3)
ax.plot(poses[:,0], poses[:,1], alpha = 0.5)
ax.plot([-3.3, 3.3], [0, 0], c = 'black') # x-axis
ax.plot([0, 0], [-3.3, 3.3], c = 'black') # y-axis
ax.plot([-3.3, 3.3], [np.cos(np.radians(60))*-3.3, np.cos(np.radians(60))*3.3], c = 'red') # y60-axis
ax.plot([-3.3, 3.3], [np.cos(np.radians(120))*-3.3, np.cos(np.radians(120))*3.3], c = 'red') # y120-axis
ax.plot([0, 0], [-3.3, 3.3], c = 'red') # y-axis

def cart2ring(x, y):

    ring1 = y

    x_component = x * np.sin(np.radians(60))
    y_component = y * np.sin(np.radians(30))

    ring2 = x_component + y_component

    x_component = x * np.sin(np.radians(120))
    y_component = y * -np.sin(np.radians(30))

    ring3 = x_component + y_component

    return np.array([ring1, ring2, ring3]).T

def ring2cart(ring1, ring2, ring3):

    ring2_x_component = ring2 / np.sin(np.radians(60))
    ring3_x_component = ring3 / np.sin(np.radians(120))

    x = (ring2_x_component + ring3_x_component) / 2
    y = ring1

    return np.array([x, y]).T

# Test points

for i in np.arange(-3, 3.5, 0.5):

    j = -1/i

    ax.plot(i, j, c = 'green', marker = '+', markersize = 30)
    ax.plot(*ring2cart(*cart2ring(i, j)), c = 'orange', marker = '+', markersize = 30)

plt.show()

# Convert datasets

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/"

data_files = [  "training_data_bl_tr/", "training_data_tr_bl/", "training_data_br_tl/", "training_data_tl_br/",
                "training_data_lm_rm/", "training_data_rm_lm/", "training_data_tm_bm/", "training_data_bm_tm/"]

fig, axes = plt.subplots(2,4)

axes = axes.flatten()

fig2, axes2 = plt.subplots(8, 3)

fig3, axes3 = plt.subplots(2, 4)

axes3 = axes3.flatten()

vel_gain = 15

neurons = [30, 40, 60, 80, 120]

sig = 10

for ring_size in neurons:
    for data_file, ax, ax2, ax3 in zip(data_files, axes, axes2, axes3):

        #x_y_data = np.loadtxt(data_folder + data_file + "body_poses.csv", delimiter = ',', skiprows = 1)[150:-150,1:3]
        x_y_data = pd.read_csv(data_folder + data_file + "body_poses.csv").iloc[150:-150, 1:3] * vel_gain

        #x_y_rings = cart2ring(x_y_data[:,0], x_y_data[:,1])
        x_y_rings = cart2ring(x_y_data["X"], x_y_data["Y"])

        #x_y_cart = ring2cart(x_y_rings[:,0], x_y_rings[:,1], x_y_rings[:,2])
        x_y_cart = ring2cart(x_y_rings[:,0], x_y_rings[:,1], x_y_rings[:,2])

        ax.set_xlim(-3.3 * vel_gain, 3.3 * vel_gain)

        ax.set_ylim(-3.3 * vel_gain, 3.3 * vel_gain)

        ax.scatter(x_y_cart[:,0], x_y_cart[:,1], c = x_y_data.index)

        # Wraparound, accounting for multiple wrap-arounds when needed (such as if ring distance is +/- many times the ring length)

        x_y_rings[x_y_rings<0] = x_y_rings[x_y_rings<0] + np.abs(x_y_rings[x_y_rings<0]//ring_size * ring_size) # + ring_size
        x_y_rings[x_y_rings>=ring_size] = x_y_rings[x_y_rings>=ring_size] - np.abs(x_y_rings[x_y_rings>=ring_size]//ring_size * ring_size) # - ring_size

        ax3.set_ylim(0, 150)

        ax3.plot(x_y_data.index, x_y_rings[:,0], alpha = 0.2)

        np.save(data_folder + data_file + "ring_indexes_{}.npy".format(ring_size), x_y_rings)

        # Create 3 blocks of indices to store ring_size gaussians, all centered on 0

        gaussian_range = np.arange(-(ring_size//2),(ring_size//2))

        x_y_ring_1_gaussians = np.resize(gaussian_range, new_shape = (len(x_y_rings), ring_size))
        x_y_ring_2_gaussians = np.resize(gaussian_range, new_shape = (len(x_y_rings), ring_size))
        x_y_ring_3_gaussians = np.resize(gaussian_range, new_shape = (len(x_y_rings), ring_size))

        # Find where the max (active grid cell) is for each ring
        max_locations_ring_1 = np.round(x_y_rings[:,0]).astype(int)
        max_locations_ring_2 = np.round(x_y_rings[:,1]).astype(int)
        max_locations_ring_3 = np.round(x_y_rings[:,2]).astype(int)

        # Create a function for a 0-mean Gaussian with the desired sigma
        pose_gaussians = norm(0, 3)

        # Apply this function onto the block of indices, giving N Gaussians all with mean 0
        zeroed_gaussians_ring_1 = pose_gaussians.pdf(x_y_ring_1_gaussians)
        zeroed_gaussians_ring_2 = pose_gaussians.pdf(x_y_ring_2_gaussians)
        zeroed_gaussians_ring_3 = pose_gaussians.pdf(x_y_ring_3_gaussians)

        # Preallocate for final Gaussians
        shifted_gaussians_ring_1 = np.empty_like(zeroed_gaussians_ring_1)
        shifted_gaussians_ring_2 = np.empty_like(zeroed_gaussians_ring_2)
        shifted_gaussians_ring_3 = np.empty_like(zeroed_gaussians_ring_3)

        # Move each Gaussian to its proper position, centred over the active grid cell
        #shifted_gaussians_ring_1 = np.roll(zeroed_gaussians_ring_1, max_locations_ring_1-(ring_size//2), axis = 1)
        #shifted_gaussians_ring_2 = np.roll(zeroed_gaussians_ring_2, max_locations_ring_2-(ring_size//2), axis = 1)
        #shifted_gaussians_ring_3 = np.roll(zeroed_gaussians_ring_3, max_locations_ring_3-(ring_size//2), axis = 1)

        # Move each Gaussian to its proper position, centred over the active grid cell
        for index in range(len(max_locations_ring_1)):
            shifted_gaussians_ring_1[index, :] = np.roll(zeroed_gaussians_ring_1[index, :], max_locations_ring_1[index]-(ring_size//2))
        for index in range(len(max_locations_ring_2)):
            shifted_gaussians_ring_2[index, :] = np.roll(zeroed_gaussians_ring_2[index, :], max_locations_ring_2[index]-(ring_size//2))
        for index in range(len(max_locations_ring_3)):
            shifted_gaussians_ring_3[index, :] = np.roll(zeroed_gaussians_ring_3[index, :], max_locations_ring_3[index]-(ring_size//2))

        # Rescale so that the Gaussians are in range 0-1
        shifted_gaussians_ring_1 = shifted_gaussians_ring_1.T * 1/np.max(shifted_gaussians_ring_1, axis = 1)
        shifted_gaussians_ring_2 = shifted_gaussians_ring_2.T * 1/np.max(shifted_gaussians_ring_2, axis = 1)
        shifted_gaussians_ring_3 = shifted_gaussians_ring_3.T * 1/np.max(shifted_gaussians_ring_3, axis = 1)

        shifted_gaussians_ring_1 = shifted_gaussians_ring_1.T
        shifted_gaussians_ring_2 = shifted_gaussians_ring_2.T
        shifted_gaussians_ring_3 = shifted_gaussians_ring_3.T

        for i in range(0, 60, 10):
            ax2[0].plot(shifted_gaussians_ring_1[i,:])
            ax2[0].vlines(max_locations_ring_1[i], 0., 1.1, color = 'grey')
            ax2[0].set_xlabel("Head Direction")
            ax2[0].set_ylabel("Pseudo-probability")
            ax2[0].set_title("Ground Truth Data")

        for i in range(0, 60, 10):
            ax2[1].plot(shifted_gaussians_ring_2[i,:])
            ax2[1].vlines(max_locations_ring_2[i], 0., 1.1, color = 'grey')
            ax2[1].set_xlabel("Head Direction")
            ax2[1].set_ylabel("Pseudo-probability")
            ax2[1].set_title("Ground Truth Data")

        for i in range(0, 60, 10):
            ax2[2].plot(shifted_gaussians_ring_3[i,:])
            ax2[2].vlines(max_locations_ring_3[i], 0., 1.1, color = 'grey')
            ax2[2].set_xlabel("Head Direction")
            ax2[2].set_ylabel("Pseudo-probability")
            ax2[2].set_title("Ground Truth Data")

        np.save(data_folder + data_file + "ring_1_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_1)
        np.save(data_folder + data_file + "ring_2_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_2)
        np.save(data_folder + data_file + "ring_3_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_3)

plt.show()
