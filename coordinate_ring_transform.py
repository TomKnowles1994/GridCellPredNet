import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace


data_filepath = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

poses = np.load(data_filepath + "filtered_body_poses.npy")[:,1:]

fig, ax = plt.subplots(1,1)

#ax.set_xlim(-3.3, 3.3)
#ax.set_ylim(-3.3, 3.3)
ax.plot(poses[:,0], poses[:,1], alpha = 0.5)
ax.plot([-3.3, 3.3], [0, 0], c = 'black') # x-axis
ax.plot([0, 0], [-3.3, 3.3], c = 'black') # y-axis
ax.plot([-3.3, 3.3], [np.cos(np.radians(60))*-3.3, np.cos(np.radians(60))*3.3], c = 'red') # y60-axis
ax.plot([-3.3, 3.3], [np.cos(np.radians(120))*-3.3, np.cos(np.radians(120))*3.3], c = 'red') # y120-axis
ax.plot([0, 0], [-3.3, 3.3], c = 'red') # y-axis

#plt.show()

def cart2ring(x, y):

    if not isinstance(x, np.ndarray):

        x = np.array(x)

    if not isinstance(y, np.ndarray):

        y = np.array(y)

    #ring1 = y.copy()

    x_component = x * np.sin(np.radians(60))
    y_component = y * np.sin(np.radians(30))

    ring2 = x_component + y_component

    x_component = x * np.sin(np.radians(120))
    y_component = y * -np.sin(np.radians(30))

    ring3 = x_component + y_component

    return np.array([y, ring2, ring3]).T

def ring2cart(ring1, ring2, ring3):

    if not isinstance(ring1, np.ndarray):

        ring1 = np.array(ring1)

    if not isinstance(ring2, np.ndarray):

        ring2 = np.array(ring2)

    if not isinstance(ring3, np.ndarray):

        ring3 = np.array(ring3)

    ring2_x = ring2 / np.sin(np.radians(60))
    ring3_x = ring3 / np.sin(np.radians(120))

    x = (ring2_x + ring3_x) / 2
    #y = ring1

    return np.array([x, ring1]).T

def wrap_to_distance(distance, boundary):
    
    wrapped = distance.copy()
    
    wrapped[distance > 0] = distance[distance > 0] % boundary
    wrapped[distance < 0] = distance[distance < 0] % boundary
    
    return wrapped

# Integrate to get ground truth ring positions

vel_gain = 15

print(poses.shape)

#vel_x = np.diff(poses[:,0])
#vel_y = np.diff(poses[:,1])

vel_x = np.gradient(poses[:,0])
vel_y = np.gradient(poses[:,1])

#vel_x, vel_y = vel_x * vel_gain, vel_y * vel_gain

velocity_magnitude = np.sqrt(vel_x ** 2 + vel_y ** 2)

velocity_angle = np.arctan2(vel_y, vel_x)

#integrated_magnitude = np.cumsum(velocity_magnitude)[::len(velocity_magnitude) // 10]

#integrated_angle = np.cumsum(velocity_angle)[::len(velocity_angle) // 10]

ring2_offset = np.radians(60)
ring3_offset = np.radians(120)

velocity_r1 = velocity_magnitude * np.sin(velocity_angle)
velocity_r2 = velocity_magnitude * np.sin(velocity_angle + ring2_offset)
velocity_r3 = velocity_magnitude * np.sin(velocity_angle + ring3_offset)

ring1_displacement = np.cumsum(velocity_r1)[::len(velocity_r1) // 10]
ring2_displacement = np.cumsum(velocity_r2)[::len(velocity_r2) // 10]
ring3_displacement = np.cumsum(velocity_r3)[::len(velocity_r3) // 10]

#ring1_displacement = np.diff(ring1_displacement)
#ring2_displacement = np.diff(ring2_displacement)
#ring3_displacement = np.diff(ring3_displacement)

#ring1_displacement = np.cumsum(ring1_displacement)[::len(velocity_r1) // 10]
#ring2_displacement = np.cumsum(ring2_displacement)[::len(velocity_r2) // 10]
#ring3_displacement = np.cumsum(ring3_displacement)[::len(velocity_r3) // 10]

#ring1_displacement = np.cumsum(velocity_r1)[::len(velocity_r1) // 10]
#ring2_displacement = np.cumsum(velocity_r2)[::len(velocity_r2) // 10]
#ring3_displacement = np.cumsum(velocity_r3)[::len(velocity_r3) // 10]

#ring1_displacement = integrated_magnitude * np.sin(integrated_angle)
#ring2_displacement = integrated_magnitude * np.cos(integrated_angle - ring2_offset)
#ring3_displacement = integrated_magnitude * np.cos(integrated_angle - ring3_offset)

derived_xy = ring2cart(ring1_displacement, ring2_displacement, ring3_displacement)

print(derived_xy)
print(derived_xy.shape)

# Test points

#for i in np.arange(-3, 3.5, 0.5):

    #j = -1/i

    #ax.plot(i, j, c = 'green', marker = '+', markersize = 30)
    #ax.plot(*ring2cart(*cart2ring(i, j)), c = 'orange', marker = '+', markersize = 30)

#ax.plot(poses[::len(poses)//10, 0], poses[::len(poses)//10, 1], c = 'green', marker = '+', markersize = 30)
ax.plot(poses[::len(poses)//10, 0], poses[::len(poses)//10, 1], c = 'green', marker = '+', markersize = 30)
ax.plot(derived_xy[:, 0], derived_xy[:, 1], c = 'orange', marker = '+', markersize = 30)

#plt.show()

# Convert translation set poses to ring codes

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/"

data_files = [  "training_data_bl_tr/", "training_data_tr_bl/", "training_data_br_tl/", "training_data_tl_br/",
                "training_data_lm_rm/", "training_data_rm_lm/", "training_data_tm_bm/", "training_data_bm_tm/"]

fig, axes = plt.subplots(2,4, sharex = True, sharey = True)

axes = axes.flatten()

fig2, axes2 = plt.subplots(8, 3, sharex = True, sharey = True)

fig3, axes3 = plt.subplots(2, 4, sharex = True, sharey = True)

axes3 = axes3.flatten()

neurons = [30, 40, 60, 80, 120]

sigmas = [neuron_count/20 for neuron_count in neurons]

for ring_size, sigma in zip(neurons, sigmas):
    for data_file, ax, ax2, ax3 in zip(data_files, axes, axes2, axes3):

        #x_y_data = np.loadtxt(data_folder + data_file + "body_poses.csv", delimiter = ',', skiprows = 1)[150:-150,1:3]
        #x_y_data = pd.read_csv(data_folder + data_file + "body_poses.csv").iloc[150:-150, 1:3] * vel_gain
        x_y_data = np.load(data_folder + data_file + "filtered_body_poses.npy")[15:-15, 1:3] * vel_gain
        x_y_data = pd.DataFrame(x_y_data, columns = ["X", "Y"])

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
        pose_gaussians = norm(0, sigma)

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
            #ax2[0].set_xlabel("Head Direction")
            #ax2[0].set_ylabel("Pseudo-probability")
            #ax2[0].set_title("Ground Truth Data")

        for i in range(0, 60, 10):
            ax2[1].plot(shifted_gaussians_ring_2[i,:])
            ax2[1].vlines(max_locations_ring_2[i], 0., 1.1, color = 'grey')
            #ax2[1].set_xlabel("Head Direction")
            #ax2[1].set_ylabel("Pseudo-probability")
            #ax2[1].set_title("Ground Truth Data")

        for i in range(0, 60, 10):
            ax2[2].plot(shifted_gaussians_ring_3[i,:])
            ax2[2].vlines(max_locations_ring_3[i], 0., 1.1, color = 'grey')
            #ax2[2].set_xlabel("Head Direction")
            #ax2[2].set_ylabel("Pseudo-probability")
            #ax2[2].set_title("Ground Truth Data")

        np.save(data_folder + data_file + "ring_1_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_1)
        np.save(data_folder + data_file + "ring_2_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_2)
        np.save(data_folder + data_file + "ring_3_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_3)

#plt.show()

# Convert rotation set poses to ring codes

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/"

data_files = [  "training_data_bottom_left/", "training_data_bottom_right/", "training_data_top_left/", "training_data_top_right/",
                "training_data_centre/", "training_data_centre_5x/", "training_data_all/"]

fig, axes = plt.subplots(1,7, sharex = True, sharey = True)

axes = axes.flatten()

fig2, axes2 = plt.subplots(7, 3, sharex = True, sharey = True)

fig3, axes3 = plt.subplots(1, 7, sharex = True, sharey = True)

axes3 = axes3.flatten()

neurons = [30, 40, 60, 80, 120]

sigmas = [neuron_count/20 for neuron_count in neurons]

for ring_size, sigma in zip(neurons, sigmas):
    for data_file, ax, ax2, ax3 in zip(data_files, axes, axes2, axes3):

        #x_y_data = np.loadtxt(data_folder + data_file + "body_poses.csv", delimiter = ',', skiprows = 1)[150:-150,1:3]
        #x_y_data = pd.read_csv(data_folder + data_file + "body_poses.csv").iloc[150:-150, 1:3] * vel_gain
        x_y_data = np.load(data_folder + data_file + "filtered_body_poses.npy")[15:-15, 1:3] * vel_gain
        x_y_data = pd.DataFrame(x_y_data, columns = ["X", "Y"])

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
        pose_gaussians = norm(0, sigma)

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
            #ax2[0].set_xlabel("Head Direction")
            #ax2[0].set_ylabel("Pseudo-probability")
            #ax2[0].set_title("Ground Truth Data")

        for i in range(0, 60, 10):
            ax2[1].plot(shifted_gaussians_ring_2[i,:])
            ax2[1].vlines(max_locations_ring_2[i], 0., 1.1, color = 'grey')
            #ax2[1].set_xlabel("Head Direction")
            #ax2[1].set_ylabel("Pseudo-probability")
            #ax2[1].set_title("Ground Truth Data")

        for i in range(0, 60, 10):
            ax2[2].plot(shifted_gaussians_ring_3[i,:])
            ax2[2].vlines(max_locations_ring_3[i], 0., 1.1, color = 'grey')
            #ax2[2].set_xlabel("Head Direction")
            #ax2[2].set_ylabel("Pseudo-probability")
            #ax2[2].set_title("Ground Truth Data")

        np.save(data_folder + data_file + "ring_1_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_1)
        np.save(data_folder + data_file + "ring_2_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_2)
        np.save(data_folder + data_file + "ring_3_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_3)

#plt.show()

# Convert rat trajectory poses to ring codes

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/"

data_file = "real_data_6005_timestamped/"

fig, ax = plt.subplots(1, 1)

fig2, ax2 = plt.subplots(1, 3, sharey = True)

fig3, ax3 = plt.subplots(1, 1)

vel_gain = 15

neurons = [30, 40, 60, 80, 120]

sigmas = [neuron_count/20 for neuron_count in neurons]

for ring_size, sigma in zip(neurons, sigmas):

    #x_y_data = np.loadtxt(data_folder + data_file + "body_poses.csv", delimiter = ',', skiprows = 1)[150:-150,1:3]
    #x_y_data = pd.read_csv(data_folder + data_file + "body_poses.csv").iloc[:, 1:3] * vel_gain
    x_y_data = np.load(data_folder + data_file + "filtered_body_poses.npy")[15:, 1:3] * vel_gain
    x_y_data = pd.DataFrame(x_y_data, columns = ["X", "Y"])

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
    pose_gaussians = norm(0, sigma)

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

def rings_to_grids(ring1, ring2, ring3, rp_window, neuron_count):

    grid_cells = np.zeros(shape = (len(ring1), rp_window ** 3))

    neurons_per_rp = neuron_count // rp_window

    #plt.plot(ring1[0, :])
    #plt.plot(ring2[0, :])
    #plt.plot(ring3[0, :])

    #plt.show()

    peak_locations_ring1 = np.argmax(ring1, axis = 1)
    peak_locations_ring2 = np.argmax(ring2, axis = 1)
    peak_locations_ring3 = np.argmax(ring3, axis = 1)

    rp_locations_ring1 = peak_locations_ring1 // neurons_per_rp
    rp_locations_ring2 = peak_locations_ring2 // neurons_per_rp
    rp_locations_ring3 = peak_locations_ring3 // neurons_per_rp

    active_grid_cells = rp_locations_ring1 + rp_locations_ring2 * rp_window + rp_locations_ring3 * rp_window ** 2

    active_grid_cells = active_grid_cells.astype(np.int)

    #for i, active_cell in enumerate(active_grid_cells):

    #    grid_cells[i, active_cell] = 1

    grid_cells[np.arange(active_grid_cells.size), active_grid_cells] = 1 # [:, index] assigns across every column, every row; [range, index] assigns to one column per row

    #plt.plot(grid_cells[0, :])

    #plt.show()

    return grid_cells

# Generate grid codes for translation sets

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/"

data_files = [  "training_data_bl_tr/", "training_data_tr_bl/", "training_data_br_tl/", "training_data_tl_br/",
                "training_data_lm_rm/", "training_data_rm_lm/", "training_data_tm_bm/", "training_data_bm_tm/",
                "training_data_all/"]

for data_file in data_files:
    for neuron_count in neurons:
        
        ring1 = np.load(data_folder + data_file + "ring_1_gaussians_{}.npy".format(neuron_count))
        ring2 = np.load(data_folder + data_file + "ring_2_gaussians_{}.npy".format(neuron_count))
        ring3 = np.load(data_folder + data_file + "ring_3_gaussians_{}.npy".format(neuron_count))

        grid_code = rings_to_grids(ring1, ring2, ring3, rp_window = 10, neuron_count = neuron_count)

        np.save(data_folder + data_file + "grid_code_{}.npy".format(neuron_count), grid_code)

        # Create a block of indices to store N gaussians, all centered on 0
        gaussian_width = grid_code.shape[1]
        gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2))
        gaussian_block = np.resize(gaussian_range, new_shape = (grid_code.shape[0], gaussian_width))

        # Find where the max (active grid cell) is for each sample
        max_locations = np.argmax(grid_code, axis = 1)

        print(max_locations.shape)
        
        # Sigma is made relative to grid cell number to ensure a reasonable spread of 'correct enough' values
        sigma = grid_code.shape[1] // 20

        # Create a function for a 0-mean Gaussian with the desired sigma
        pose_gaussians = norm(0, sigma)

        # Apply this function onto the block of indices, giving N Gaussians all with mean 0
        zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

        # Preallocate for final Gaussians
        shifted_gaussians = np.empty_like(zeroed_gaussians)

        # Move each Gaussian to its proper position, centred over the active grid cell
        for index in range(len(max_locations)):
            shifted_gaussians[index, :] = np.roll(zeroed_gaussians[index, :], max_locations[index]-(grid_code.shape[1]//2))

        # Rescale so that the Gaussians are in range 0-1
        scaling_factor = 1/np.max(shifted_gaussians, axis = 1)
        shifted_gaussians = shifted_gaussians * scaling_factor[:, None] # Saves doing 2 transposes

        np.save(data_folder + data_file + "gaussian_grid_code_{}.npy".format(neuron_count), shifted_gaussians)

plt.plot(grid_code[0, :])

plt.plot(shifted_gaussians[0, :])

plt.show()

# Generate grid codes for rotation sets

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/"

data_files = [  "training_data_bottom_left/", "training_data_bottom_right/", "training_data_top_left/", "training_data_top_right/",
                "training_data_centre/", "training_data_centre_5x/", "training_data_all/"]

for data_file in data_files:
    for neuron_count in neurons:
        
        ring1 = np.load(data_folder + data_file + "ring_1_gaussians_{}.npy".format(neuron_count))
        ring2 = np.load(data_folder + data_file + "ring_2_gaussians_{}.npy".format(neuron_count))
        ring3 = np.load(data_folder + data_file + "ring_3_gaussians_{}.npy".format(neuron_count))

        grid_code = rings_to_grids(ring1, ring2, ring3, rp_window = 10, neuron_count = neuron_count)

        np.save(data_folder + data_file + "grid_code_{}.npy".format(neuron_count), grid_code)

        # Create a block of indices to store N gaussians, all centered on 0
        gaussian_width = grid_code.shape[1]
        gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2))
        gaussian_block = np.resize(gaussian_range, new_shape = (grid_code.shape[0], gaussian_width))

        # Find where the max (active grid cell) is for each sample
        max_locations = np.argmax(grid_code, axis = 1)

        print(max_locations.shape)
        
        # Sigma is made relative to grid cell number to ensure a reasonable spread of 'correct enough' values
        sigma = grid_code.shape[1] // 20

        # Create a function for a 0-mean Gaussian with the desired sigma
        pose_gaussians = norm(0, sigma)

        # Apply this function onto the block of indices, giving N Gaussians all with mean 0
        zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

        # Preallocate for final Gaussians
        shifted_gaussians = np.empty_like(zeroed_gaussians)

        # Move each Gaussian to its proper position, centred over the active grid cell
        for index in range(len(max_locations)):
            shifted_gaussians[index, :] = np.roll(zeroed_gaussians[index, :], max_locations[index]-(grid_code.shape[1]//2))

        # Rescale so that the Gaussians are in range 0-1
        scaling_factor = 1/np.max(shifted_gaussians, axis = 1)
        shifted_gaussians = shifted_gaussians * scaling_factor[:, None] # Saves doing 2 transposes

        np.save(data_folder + data_file + "gaussian_grid_code_{}.npy".format(neuron_count), shifted_gaussians)

plt.plot(grid_code[0, :])

plt.plot(shifted_gaussians[0, :])

plt.show()

# Generate grid codes for rat trajectory

data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/"

data_file = "real_data_6005_timestamped/"

for neuron_count in neurons:
    
    ring1 = np.load(data_folder + data_file + "ring_1_gaussians_{}.npy".format(neuron_count))
    ring2 = np.load(data_folder + data_file + "ring_2_gaussians_{}.npy".format(neuron_count))
    ring3 = np.load(data_folder + data_file + "ring_3_gaussians_{}.npy".format(neuron_count))

    grid_code = rings_to_grids(ring1, ring2, ring3, rp_window = 10, neuron_count = neuron_count)

    np.save(data_folder + data_file + "grid_code_{}.npy".format(neuron_count), grid_code)

    # Create a block of indices to store N gaussians, all centered on 0
    gaussian_width = grid_code.shape[1]
    gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2))
    gaussian_block = np.resize(gaussian_range, new_shape = (grid_code.shape[0], gaussian_width))

    # Find where the max (active grid cell) is for each sample
    max_locations = np.argmax(grid_code, axis = 1)

    print(max_locations.shape)
    
    # Sigma is made relative to grid cell number to ensure a reasonable spread of 'correct enough' values
    sigma = grid_code.shape[1] // 20

    # Create a function for a 0-mean Gaussian with the desired sigma
    pose_gaussians = norm(0, sigma)

    # Apply this function onto the block of indices, giving N Gaussians all with mean 0
    zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

    # Preallocate for final Gaussians
    shifted_gaussians = np.empty_like(zeroed_gaussians)

    # Move each Gaussian to its proper position, centred over the active grid cell
    for index in range(len(max_locations)):
        shifted_gaussians[index, :] = np.roll(zeroed_gaussians[index, :], max_locations[index]-(grid_code.shape[1]//2))

    # Rescale so that the Gaussians are in range 0-1
    scaling_factor = 1/np.max(shifted_gaussians, axis = 1)
    shifted_gaussians = shifted_gaussians * scaling_factor[:, None] # Saves doing 2 transposes

    np.save(data_folder + data_file + "gaussian_grid_code_{}.npy".format(neuron_count), shifted_gaussians)

plt.plot(grid_code[0, :])

plt.plot(shifted_gaussians[0, :])

plt.show()