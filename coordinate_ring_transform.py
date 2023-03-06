import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, laplace

data_filepath = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

poses = np.load(data_filepath + "filtered_body_poses.npy")[:,1:]

sharpness = 40

vel_gain = 15

offset = 7 # degrees of r1 (and by extension, the entire ring axes system) rotated anticlockwise from y

grid_format = 'gaussian'    

# One of:-

# 'index': ring indexes only, shape = (?, 3*len(neurons))
# 'gaussian': a gaussian centred over 0-999 depending on ring index combination, shape = (?, rp_window ** 3)

#neurons = [30, 40, 60, 80, 120]

neurons = [40, 50, 60, 70]

sigmas = [neuron_count/sharpness for neuron_count in neurons]

def cart2ring(x, y, offset):

    if not isinstance(x, np.ndarray):

        x = np.array(x)

    if not isinstance(y, np.ndarray):

        y = np.array(y)

    assert x.size == y.size
    assert x.size > 0

    if x.size == 1:

        projection_r1 = np.dot(np.array([x, y]), np.array([x, -x / np.tan(np.radians(offset))])) / np.linalg.norm(np.array([x, -x / np.tan(np.radians(offset))]))
        #print("Projection r_1: {}".format(projection_r1))
        projection_r2 = np.dot(np.array([x, y]), np.array([x, -x / np.tan(np.radians(60 + offset))])) / np.linalg.norm(np.array([x, -x / np.tan(np.radians(60 + offset))]))
        #print("Projection r_2: {}".format(projection_r2))
        projection_r3 = np.dot(np.array([x, y]), np.array([x, x / np.tan(np.radians(60 - offset))])) / np.linalg.norm(np.array([x, x / np.tan(np.radians(60 - offset))]))
        #print("Projection r_3: {}".format(projection_r3))

    elif x.size > 1:

        projection_r1 = np.array([np.dot(np.array([i, j]), np.array([i, -i / np.tan(np.radians(offset))])) / np.linalg.norm(np.array([i, -i / np.tan(np.radians(offset))])) for i, j in zip(x, y)])
        #print("Projection r_1: {}".format(projection_r1))
        projection_r2 = np.array([np.dot(np.array([i, j]), np.array([i, -i / np.tan(np.radians(60 + offset))])) / np.linalg.norm(np.array([i, -i / np.tan(np.radians(60 + offset))])) for i, j in zip(x, y)])
        #print("Projection r_2: {}".format(projection_r2))
        projection_r3 = np.array([np.dot(np.array([i, j]), np.array([i, i / np.tan(np.radians(60 - offset))])) / np.linalg.norm(np.array([i, i / np.tan(np.radians(60 - offset))])) for i, j in zip(x, y)])
        #print("Projection r_3: {}".format(projection_r3))

    else:

        raise NotImplementedError

    ring1 = -np.sign(x) * projection_r1
    ring2 = -np.sign(x) * projection_r2
    ring3 = -np.sign(x) * projection_r3

    # print(np.array([ring1, ring2, ring3]).T.shape)
    # print(np.array([ring1, ring2, ring3]).shape)
    # print(*np.array([ring1, ring2, ring3]).T)
    # print(*np.array([ring1, ring2, ring3]))

    return np.array([ring1, ring2, ring3]).T

def ring2cart(ring1, ring2, ring3, offset):

    if not isinstance(ring1, np.ndarray):

        ring1 = np.array(ring1)

    if not isinstance(ring2, np.ndarray):

        ring2 = np.array(ring2)

    if not isinstance(ring3, np.ndarray):

        ring3 = np.array(ring3)

    assert ring1.size > 0

    ### New method: Use intersection of normals to find the corresponding (x,y) 
    
    # Draw ring axes that span the length of the arena in question

    # x = np.array([i for i in np.linspace(np.min(ring1 * np.cos(ring1)), np.max(ring1 * np.cos(ring1)), 0.01)])
    # y = np.array([i for i in np.linspace(np.min(ring2 * np.cos(ring2)), np.max(ring2 * np.cos(ring2)), 0.01)])

    # The maximum extent of x and y are equal to the longest ring
    # The largest ratio of ring:cartesian values are if a ring axis is aligned exactly to x or y
    # Therefore, no point in ring space can be outside the corresponding bounds in cartesian space

    max_x = np.max([ring1, ring2, ring3], axis = 0) / np.cos(np.radians(np.max([offset, offset + 60, offset + 120])))
    min_x = -max_x

    ring1_y = ring1 / np.cos(np.radians(offset))
    ring2_y = ring2 / np.cos(np.radians(60 + offset))
    ring3_y = -ring3 / np.cos(np.radians(60 - offset))

    y_r1_n_start    =  -max_x * np.tan(np.radians(offset)) + ring1_y
    y_r1_n_end      =  -min_x * np.tan(np.radians(offset)) + ring1_y

    #ax.plot([min_x, max_x], [y_r1_n_start, y_r1_n_end], color = 'blue')

    y_r2_n_start    =  -max_x * np.tan(np.radians(60 + offset)) + ring2_y
    y_r2_n_end      =  -min_x * np.tan(np.radians(60 + offset)) + ring2_y

    #ax.plot([min_x, max_x], [y_r2_n_start, y_r2_n_end], color = 'blue')

    y_r3_n_start    =  max_x * np.tan(np.radians(60 - offset)) + ring3_y
    y_r3_n_end      =  min_x * np.tan(np.radians(60 - offset)) + ring3_y
    
    #ax.plot([min_x, max_x], [y_r3_n_start, y_r3_n_end], color = 'blue')

    # Get start and end points of ring axes

    # start_r1_n = np.array([x[0], y_r1_n[0]])
    # start_r2_n = np.array([x[0], y_r2_n[0]])
    # start_r3_n = np.array([x[0], y_r3_n[0]])

    # end_r1_n = np.array([x[-1], y_r1_n[-1]])
    # end_r2_n = np.array([x[-1], y_r2_n[-1]])
    # end_r3_n = np.array([x[-1], y_r3_n[-1]])

    start_r1_n = np.array([min_x, y_r1_n_start])
    start_r2_n = np.array([min_x, y_r2_n_start])
    start_r3_n = np.array([min_x, y_r3_n_start])

    end_r1_n = np.array([max_x, y_r1_n_end])
    end_r2_n = np.array([max_x, y_r2_n_end])
    end_r3_n = np.array([max_x, y_r3_n_end])

    # Calculate where each pair intersects

    # These *should* be approximately the same, unless offset aligns a ring with x or y

    x_values = np.empty(shape = (ring1.size, 3))
    y_values = np.empty(shape = (ring1.size, 3))

    x1,y1 = start_r1_n
    x2,y2 = end_r1_n
    x3,y3 = start_r2_n
    x4,y4 = end_r2_n

    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    x = x3 + ua * (x4-x3)
    y = y3 + ua * (y4-y3)

    x_values[:, 0] = x
    y_values[:, 0] = y

    x1,y1 = start_r2_n
    x2,y2 = end_r2_n
    x3,y3 = start_r3_n
    x4,y4 = end_r3_n

    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    x = x3 + ua * (x4-x3)
    y = y3 + ua * (y4-y3)

    x_values[:, 1] = x
    y_values[:, 1] = y

    x1,y1 = start_r1_n
    x2,y2 = end_r1_n
    x3,y3 = start_r3_n
    x4,y4 = end_r3_n

    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    x = x3 + ua * (x4-x3)
    y = y3 + ua * (y4-y3)

    x_values[:, 2] = x
    y_values[:, 2] = y

    x = np.squeeze(np.mean(np.array(x_values), axis = 1))
    y = np.squeeze(np.mean(np.array(y_values), axis = 1))

    # print(np.array([x, y]).T.shape)
    # print(np.array([x, y]).shape)
    # print(*np.array([x, y]).T)
    # print(*np.array([x, y]))

    return np.array([x, y]).T

def wrap_to_distance(distance, boundary):
    
    wrapped = distance.copy()
    
    wrapped[distance > 0] = distance[distance > 0] % boundary
    wrapped[distance < 0] = distance[distance < 0] % boundary
    
    return wrapped

def generate_ring_codes(data_folder, data_files, offset, vel_gain = 1, plot = False, data_type = None):

    # if plot and data_type is None:

    #     raise ValueError

    # if plot and data_type == 'rotation':

    #     fig, axes = plt.subplots(1,7, sharex = True, sharey = True)

    #     axes = axes.flatten()

    #     fig2, axes2 = plt.subplots(7, 3, sharex = True, sharey = True)

    #     fig3, axes3 = plt.subplots(1, 7, sharex = True, sharey = True)

    #     axes3 = axes3.flatten()
    
    # if plot and data_type == 'translation':

    #     fig, axes = plt.subplots(2,4, sharex = True, sharey = True)

    #     axes = axes.flatten()

    #     fig2, axes2 = plt.subplots(8, 3, sharex = True, sharey = True)

    #     fig3, axes3 = plt.subplots(2, 4, sharex = True, sharey = True)

    #     axes3 = axes3.flatten()

    #     fig4, axes4 = plt.subplots(2, 4, sharex = True, sharey = True)

    #     axes4 = axes4.flatten()

    #     fig5, axes5 = plt.subplots(2, 4, sharex = True, sharey = True)

    #     axes5 = axes5.flatten()

    # if plot and data_type == 'rat':

    #     fig, ax = plt.subplots(1, 1)

    #     fig2, ax2 = plt.subplots(1, 3, sharey = True)

    #     fig3, ax3 = plt.subplots(1, 1)

    for ring_size, sigma in zip(neurons, sigmas):
        #for data_file, ax, ax2, ax3, ax4, ax5 in zip(data_files, axes, axes2, axes3, axes4, axes5):
        for data_file in data_files:

            print("Generating ring codes for {}".format(data_file))

            x_y_data = np.load(data_folder + data_file + "filtered_body_poses.npy")[:, 1:3] * vel_gain
            x_y_data = pd.DataFrame(x_y_data, columns = ["X", "Y"])

            x_y_rings = cart2ring(x_y_data["X"], x_y_data["Y"], offset)

            x_y_cart = ring2cart(x_y_rings[:,0], x_y_rings[:,1], x_y_rings[:,2], offset)

            # if plot:

            #     axes[0].set_xlim(-3.3 * vel_gain, 3.3 * vel_gain)

            #     ax.set_ylim(-3.3 * vel_gain, 3.3 * vel_gain)

            #     ax.scatter(x_y_cart[:,0], x_y_cart[:,1], c = x_y_data.index)

            #     ax.set_title(data_file[14:-1])

            # Wraparound

            x_y_rings[x_y_rings<0] = x_y_rings[x_y_rings<0] % ring_size
            x_y_rings[x_y_rings>=ring_size] = x_y_rings[x_y_rings>=ring_size] % ring_size


            #ax3.set_ylim(0, 150)

            # if plot:

            #     ax3.scatter(x_y_data.index, x_y_rings[:,0], alpha = 0.02)

            #     ax3.set_title(data_file[14:-1] + " ring1")

            #     ax4.scatter(x_y_data.index, x_y_rings[:,1], alpha = 0.02)

            #     ax4.set_title(data_file[14:-1] + " ring2")

            #     ax5.scatter(x_y_data.index, x_y_rings[:,2], alpha = 0.02)

            #     ax5.set_title(data_file[14:-1] + " ring3")

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

            # if plot:

            #     for i in range(0, 60, 10):
            #         ax2[0].plot(shifted_gaussians_ring_1[i,:])
            #         ax2[0].vlines(max_locations_ring_1[i], 0., 1.1, color = 'grey')
            #         #ax2[0].set_xlabel("Head Direction")
            #         #ax2[0].set_ylabel("Pseudo-probability")
            #         #ax2[0].set_title("Ground Truth Data")

            #     for i in range(0, 60, 10):
            #         ax2[1].plot(shifted_gaussians_ring_2[i,:])
            #         ax2[1].vlines(max_locations_ring_2[i], 0., 1.1, color = 'grey')
            #         #ax2[1].set_xlabel("Head Direction")
            #         #ax2[1].set_ylabel("Pseudo-probability")
            #         #ax2[1].set_title("Ground Truth Data")

            #     for i in range(0, 60, 10):
            #         ax2[2].plot(shifted_gaussians_ring_3[i,:])
            #         ax2[2].vlines(max_locations_ring_3[i], 0., 1.1, color = 'grey')
            #         #ax2[2].set_xlabel("Head Direction")
            #         #ax2[2].set_ylabel("Pseudo-probability")
            #         #ax2[2].set_title("Ground Truth Data")

            np.save(data_folder + data_file + "ring_1_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_1)
            np.save(data_folder + data_file + "ring_2_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_2)
            np.save(data_folder + data_file + "ring_3_gaussians_{}.npy".format(ring_size), shifted_gaussians_ring_3)

    if plot:

        plt.show()

def rings_to_grids(ring1, ring2, ring3, rp_window, neuron_count, grid_format = "gaussian"):

    grid_cells = np.zeros(shape = (len(ring1), rp_window ** 3))

    neurons_per_rp = neuron_count // rp_window

    plt.show()

    peak_locations_ring1 = np.argmax(ring1, axis = 1)
    peak_locations_ring2 = np.argmax(ring2, axis = 1)
    peak_locations_ring3 = np.argmax(ring3, axis = 1)

    rp_locations_ring1 = peak_locations_ring1 // neurons_per_rp
    rp_locations_ring2 = peak_locations_ring2 // neurons_per_rp
    rp_locations_ring3 = peak_locations_ring3 // neurons_per_rp

    if grid_format == 'gaussian':

        active_grid_cells = rp_locations_ring1 + rp_locations_ring2 * rp_window + rp_locations_ring3 * rp_window ** 2

    elif grid_format == 'index':

        active_grid_cells = np.array([rp_locations_ring1, rp_locations_ring2, rp_locations_ring3]).T

    #print("Active grid cells shape = {}".format(active_grid_cells.shape))

    active_grid_cells = active_grid_cells.astype(np.int)

    if grid_format == 'gaussian':

        grid_cells[np.arange(active_grid_cells.size), active_grid_cells] = 1 # [:, index] assigns across every column, every row; [range, index] assigns to one column per row

    elif grid_format == 'index':

        grid_cells = active_grid_cells

    return grid_cells

def generate_grid_codes(data_folder, data_files, neuron_count, plot = False):

    for data_file in data_files:

        print("Generating grid codes for {}".format(data_file))

        for neuron_count in neurons:
            
            ring1 = np.load(data_folder + data_file + "ring_1_gaussians_{}.npy".format(neuron_count))
            ring2 = np.load(data_folder + data_file + "ring_2_gaussians_{}.npy".format(neuron_count))
            ring3 = np.load(data_folder + data_file + "ring_3_gaussians_{}.npy".format(neuron_count))

            # print("Ring 1 Shape: {}".format(ring1.shape))
            # print("Ring 2 Shape: {}".format(ring2.shape))
            # print("Ring 3 Shape: {}".format(ring3.shape))

            grid_code = rings_to_grids(ring1, ring2, ring3, rp_window = 10, neuron_count = neuron_count)

            np.save(data_folder + data_file + "grid_code_{}.npy".format(neuron_count), grid_code)

            # Create a block of indices to store N gaussians, all centered on 0
            gaussian_width = grid_code.shape[1]
            gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2))
            gaussian_block = np.resize(gaussian_range, new_shape = (grid_code.shape[0], gaussian_width))

            # Find where the max (active grid cell) is for each sample
            max_locations = np.argmax(grid_code, axis = 1)

            # Sigma is made relative to grid cell number to ensure a reasonable spread of 'correct enough' values
            sigma = grid_code.shape[1] / sharpness

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

            np.save(data_folder + data_file + "{}_grid_code_{}.npy".format(grid_format, neuron_count), shifted_gaussians)

        if plot:

            plt.plot(grid_code[0, :])

            plt.plot(shifted_gaussians[0, :])

            plt.show()

def generate_place_codes(data_folder, data_files, neuron_count, plot = False, animate = False):

    if plot:

        fig, ax = plt.subplots(len(neurons)+1, 1)

    for data_file in data_files:

        print("Generating place codes for {}".format(data_file))

        grids = []

        for i, neuron_count in enumerate(neurons):
            
            grids.append(np.load(data_folder + data_file + "{}_grid_code_{}.npy".format(grid_format, neuron_count)))

            if plot:

                ax[i].plot(np.arange(0, 1000), grids[-1][-1])

        place_code = np.sum(grids, axis = 0) / len(neurons)

        np.save(data_folder + data_file + "{}_place_code.npy".format(grid_format, neuron_count), place_code)

        if plot:

            ax[-1].plot(np.arange(0, 1000), place_code[-1])

            plt.show()

        

        if animate:

            fig, ax = plt.subplots(len(neurons) + 1, 1, sharex = True)

            g1, = ax[0].plot(np.arange(0, 1000), grids[0][0])
            g2, = ax[1].plot(np.arange(0, 1000), grids[1][0])
            g3, = ax[2].plot(np.arange(0, 1000), grids[2][0])
            g4, = ax[3].plot(np.arange(0, 1000), grids[3][0])
            pc, = ax[4].plot(np.arange(0, 1000), place_code[0])
            text = ax[-1].text(0, 0, "")

            def animate_place_code(i):

                g1.set_data(np.arange(0, 1000), grids[0][i])
                g2.set_data(np.arange(0, 1000), grids[1][i])
                g3.set_data(np.arange(0, 1000), grids[2][i])
                g4.set_data(np.arange(0, 1000), grids[3][i])
                pc.set_data(np.arange(0, 1000), place_code[i])
                text.set_text("{}/{}".format(i+1, len(grids[0])))

                return g1, g2, g3, g4, pc
        
            ani = FuncAnimation(fig, animate_place_code, frames = len(grids[0]), interval = 100)

            plt.show()

def test_plot():

    x_start = 3.3
    x_end = -3.3

    for i in np.arange(-3, 3.5, 0.5):

        j = -1/i

        if i != 0:

            fig, ax = plt.subplots(1,1, figsize = (6,6))

            ax.set_xlim(-3.3, 3.3)
            ax.set_ylim(-3.3, 3.3)

            input_x, input_y = i, j
            r1, r2, r3 = cart2ring(i, j, offset = offset)
            output_x, output_y = ring2cart(*cart2ring(i, j, offset = offset).T, offset = offset)

            ax.plot(poses[:,0], poses[:,1], alpha = 0.5)
            ax.plot([-3.3, 3.3], [0, 0], c = 'black') # x-axis
            ax.plot([0, 0], [-3.3, 3.3], c = 'black') # y-axis
            ax.plot([x_start, x_end], [-x_start/np.tan(np.radians(offset)), -x_end/np.tan(np.radians(offset))], c = 'red', label = 'Ring axes (offset {}\N{DEGREE SIGN})'.format(offset)) # r1
            ax.plot([x_start, x_end], [-x_start/np.tan(np.radians(60 + offset)), -x_end/np.tan(np.radians(60 + offset))], c = 'red') # r2
            ax.plot([x_start, x_end], [x_start/np.tan(np.radians(60 - offset)), x_end/np.tan(np.radians(60 - offset))], c = 'red') # r3
            
            r1_y = r1 / np.cos(np.radians(offset))
            r2_y = r2 / np.cos(np.radians(60 + offset))
            r3_y = -r3 / np.cos(np.radians(60 - offset))
            
            ax.plot([x_start, x_end], [x_start*np.tan(np.radians(offset)) + r1_y, x_end*np.tan(np.radians(offset)) + r1_y], c = 'blue', linestyle = '--', linewidth = 1, label = 'Ring indices') # r1
            ax.plot([x_start, x_end], [x_start*np.tan(np.radians(60 + offset)) + r2_y, x_end*np.tan(np.radians(60 + offset)) + r2_y], c = 'blue', linestyle = '--', linewidth = 1) # r2
            ax.plot([x_start, x_end], [-x_start*np.tan(np.radians(60 - offset)) + r3_y, -x_end*np.tan(np.radians(60 - offset)) + r3_y], c = 'blue', linestyle = '--', linewidth = 1) # r3
            print("Input = {}, {}".format(i, j))
            print("Ring Values = {}, {}, {}".format(r1, r2, r3))
            print("Converted XY = {}, {}".format(output_x, output_y))
            ax.plot(i, j, c = 'green', marker = '+', markersize = 30, label = 'Cartesian input')
            ax.plot(output_x, output_y, c = 'purple', marker = '+', markersize = 20, label = 'Cartesian output')
            ax.legend(loc = 'lower right')
            plt.show()

    fig, ax = plt.subplots(1,1, figsize = (6,6))

    ax.plot(*ring2cart(*cart2ring(poses[::len(poses)//10, 0], poses[::len(poses)//10, 1], offset = offset).T, offset = offset).T, c = 'red', alpha = 0.75, marker = '+', markersize = 30, label = 'Transformed AA-1')
    ax.plot(poses[::len(poses)//10, 0] + 0.02, poses[::len(poses)//10, 1], c = 'blue', alpha = 0.75, marker = '+', markersize = 30, label = 'Ground Truth Samples')
    #ax.plot(derived_xy[:, 0], derived_xy[:, 1], c = 'orange', marker = '+', markersize = 30, label = 'Transformed RA-1')

    fig.legend()

    plt.show()

# Integrate to get ground truth ring positions

#vel_x = np.diff(poses[:,0])
#vel_y = np.diff(poses[:,1])

vel_x = np.gradient(poses[:,0])
vel_y = np.gradient(poses[:,1])

#vel_x, vel_y = vel_x * vel_gain, vel_y * vel_gain

velocity_magnitude = np.sqrt(vel_x ** 2 + vel_y ** 2)

velocity_angle = np.arctan2(vel_y, vel_x)
ring2_offset = np.radians(60)
ring3_offset = np.radians(120)

velocity_r1 = velocity_magnitude * np.sin(velocity_angle)
velocity_r2 = velocity_magnitude * np.sin(velocity_angle + ring2_offset)
velocity_r3 = velocity_magnitude * np.sin(velocity_angle + ring3_offset)

ring1_displacement = np.cumsum(velocity_r1)[::len(velocity_r1) // 10]
ring2_displacement = np.cumsum(velocity_r2)[::len(velocity_r2) // 10]
ring3_displacement = np.cumsum(velocity_r3)[::len(velocity_r3) // 10]

derived_xy = ring2cart(ring1_displacement, ring2_displacement, ring3_displacement, offset = offset)

# Test points

#test_plot()

# Original (Guifen-like) arena

# Convert translation set poses to ring and grid codes

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/"

data_files = [  "training_data_bl_tr/", "training_data_tr_bl/", "training_data_br_tl/", "training_data_tl_br/",
                "training_data_lm_rm/", "training_data_rm_lm/", "training_data_tm_bm/", "training_data_bm_tm/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'translation')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Convert rotation set poses to ring and grid codes

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_rotation_sets_3000/"

data_files = [  "training_data_bottom_left/", "training_data_bottom_right/", "training_data_top_left/", "training_data_top_right/",
                "training_data_centre/", "training_data_centre_5x/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'rotation')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Convert rat trajectory poses to ring codes

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/"

data_files = [  "real_data_6005_timestamped/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'rat')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Generate ring and grid codes for merged (rotating and translation) dataset

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/"

data_files = [  "training_data_vis_hd_grid/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False)

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)


# Dome arena

# Generate ring and grid codes for translation sets

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_translation_sets_3000/"

data_files = [  "training_dome_bl_tr/", "training_dome_tr_bl/", "training_dome_br_tl/", "training_dome_tl_br/",
                "training_dome_lm_rm/", "training_dome_rm_lm/", "training_dome_tm_bm/", "training_dome_bm_tm/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'translation')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Generate ring and grid codes for rotation sets

# Missing 'centre' dataset

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/training_dome_rotation_sets_3000/"

data_files = [  "training_dome_bottom_left/", "training_dome_bottom_right/", "training_dome_top_left/", "training_dome_top_right/",
                "training_dome_centre_5x/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'rotation')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Generate ring and grid codes for rat trajectory

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/"

data_files = [  "real_dome_6005_timestamped/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'rat')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Generate ring and grid codes for merged (rotating and translation) dataset

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_dome/"

data_files = [  "training_dome_vis_hd_grid/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False)

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)


# Oracle arena

# Convert translation set poses to ring and grid codes

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/"

data_files = [  "bl_tr/", "tr_bl/", "br_tl/", "tl_br/", "lm_rm/", "rm_lm/", "tm_bm/", "bm_tm/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'translation')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Convert rotation set poses to ring and grid codes

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/training_data/"

data_files = [  "bottom_left/", "bottom_right/", "top_left/", "top_right/", "centre/", "centre_5x/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'rotation')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = False)

# Convert rat trajectory poses to ring codes

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/"

data_files= [   "real_rat/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False, data_type = 'rat')

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = True)

# Generate ring and grid codes for merged (rotating and translation) dataset

data_folder =   "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_oracle_datasets/"

data_files= [   "training_data_vis_hd_grid/"]

generate_ring_codes(data_folder = data_folder, data_files = data_files, offset = offset, vel_gain = vel_gain, plot = False)

generate_grid_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False)

generate_place_codes(data_folder = data_folder, data_files = data_files, neuron_count = neurons, plot = False, animate = True)
