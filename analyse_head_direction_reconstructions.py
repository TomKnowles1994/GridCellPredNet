import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.animation as ani

#data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

#data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_centre/"
data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

#data = np.load(data_folder + 'networkOutput_gaussianised.npy')[3000:3050] # 2500 Test Samples

data = np.load(data_folder + 'networkOutput_gaussianised.npy')[:290]

#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_6005_timestamped_reconstructions_no_visual\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_representations_no_grid\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\whiskeye_guifen_real_data_6005_timestamped_Gc5_VisHd\\no_hd\\"
#reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_centre/training_data_centre/visual/"
reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_rat_centre/real_data_6005_timestamped/visual/"
#reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_rat_all/real_data_6005_timestamped/visual/"

#reconstructions = np.load(reconstructions_folder + 'reconstructions_hd_cells.npy')
reconstructions = np.load(reconstructions_folder + 'reconstructions_head_direction.npy')

reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_rat_all/real_data_6005_timestamped/visual/"

reconstructions_2 = np.load(reconstructions_folder + 'reconstructions_head_direction.npy')[:290]

print(data.shape)
print(reconstructions.shape)

non_zero_rows = np.unique(np.nonzero(data)[0])

# for i in range(0, 100, 10):

#     plt.plot(np.arange(data.shape[1]), data[i,:])
#     plt.plot(np.arange(data.shape[1]), reconstructions[i,:])
#     plt.show()

fig, ax = plt.subplots(5,1, gridspec_kw={'height_ratios':[3,1,1,1,1]}, figsize = (20, 15))

plt.subplots_adjust(left=0.125, bottom=None, right=0.9, top=None, wspace=None, hspace=None)

a1 = ax[0].plot(np.arange(data.shape[1]), data[0,:], label = 'Test Data')[0]
a2 = ax[0].plot(np.arange(data.shape[1]), reconstructions[0,:], label = 'Reconstruction (Centre)')[0]
a3 = ax[0].axvline(np.argmax(reconstructions[0,:]), 0, 1, color = 'purple', label = 'Reconstruction Peak (Centre)')
a4 = ax[0].text(0, 0.02, "{}/{}".format(0, len(data)))
a5 = ax[0].plot(np.arange(data.shape[1]), reconstructions_2[0,:], label = 'Reconstruction (All)')[0]
a6 = ax[0].axvline(np.argmax(reconstructions_2[0,:]), 0, 1, color = 'red', label = 'Reconstruction Peak (All)')
ax[0].set_title("Train: Centre Perspective Rotation; Test: Real Rat")

mse = np.mean((data - reconstructions) ** 2, axis = 1)
mse_2 = np.mean((data - reconstructions_2) ** 2, axis = 1)
distance_from_max_0 = np.abs(np.argmax(data, axis = 1) - np.argmax(reconstructions, axis = 1))
distance_from_max_1 = np.abs((np.argmax(data, axis = 1) + 180) - np.argmax(reconstructions, axis = 1))
distance_from_max_2 = np.abs((np.argmax(data, axis = 1) - 180) - np.argmax(reconstructions, axis = 1))
distance_from_max = np.min([distance_from_max_0, distance_from_max_1, distance_from_max_2], axis = 0)
distance_from_max_2_0 = np.abs(np.argmax(data, axis = 1) - np.argmax(reconstructions_2, axis = 1))
distance_from_max_2_1 = np.abs((np.argmax(data, axis = 1) + 180) - np.argmax(reconstructions_2, axis = 1))
distance_from_max_2_2 = np.abs((np.argmax(data, axis = 1) - 180) - np.argmax(reconstructions_2, axis = 1))
distance_from_max_2 = np.min([distance_from_max_2_0, distance_from_max_2_1, distance_from_max_2_2], axis = 0)
roughness = np.abs(np.mean(np.diff(reconstructions, axis = 1), axis = 1))
roughness_2 = np.abs(np.mean(np.diff(reconstructions_2, axis = 1), axis = 1))
theta = np.argmax(data, axis = 1)

b1 = ax[1].plot(np.arange(data.shape[0]), mse, label = 'Mean Squared Error (Centre)')[0]
b2 = ax[1].text(0, 0.01, "Sum (C): {}".format(np.round(mse[0], 2)))
b3 = ax[1].plot(np.arange(data.shape[0]), mse_2, label = 'Mean Squared Error (All)')[0]
b4 = ax[1].text(250, 0.01, "Sum (A): {}".format(np.round(mse_2[0], 2)))
c1 = ax[2].plot(np.arange(data.shape[0]), distance_from_max, label = 'Distance from Max (Centre)')[0]
c2 = ax[2].text(0, 1, "Sum (C): {}".format(distance_from_max[0]))
c3 = ax[2].plot(np.arange(data.shape[0]), distance_from_max_2, label = 'Distance from Max (All)')[0]
c4 = ax[2].text(250, 1, "Sum (A): {}".format(distance_from_max_2[0]))
d1 = ax[3].plot(np.arange(data.shape[0]), roughness, label = 'Roughness (Centre)')[0]
d2 = ax[3].text(0, 0.02, "Sum (C): {}".format(roughness[0]))
d3 = ax[3].plot(np.arange(data.shape[0]), roughness_2, label = 'Roughness (All)')[0]
d4 = ax[3].text(250, 0.02, "Sum (A): {}".format(roughness_2[0]))
e1 = ax[4].plot(np.arange(data.shape[0]), theta, label = 'Active Neuron')[0]
e2 = ax[4].text(0, 0.02, "{}".format(theta[0] % 2*np.pi))

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()


ax[1].set_xlim(0, data.shape[0])
ax[1].set_ylim(0, np.max(mse))
ax[2].set_xlim(0, data.shape[0])
ax[2].set_ylim(0, np.max(distance_from_max))
ax[3].set_xlim(0, data.shape[0])
ax[3].set_ylim(0, np.max(roughness))
ax[4].set_xlim(0, data.shape[0])
ax[4].set_ylim(0, 180)

def animate(i):

    a1.set_data(np.arange(data.shape[1]), data[i,:])
    a2.set_data(np.arange(data.shape[1]), reconstructions[i,:])
    a3.set_data(np.argmax(reconstructions[i,:]), [0,1])
    a4.set_text("{}/{}".format(i, len(data)))
    a5.set_data(np.arange(data.shape[1]), reconstructions_2[i,:])
    a6.set_data(np.argmax(reconstructions_2[i,:]), [0,1])

    # ax[1].scatter(i, mse[i])
    # ax[2].scatter(i, distance_from_max[i])
    # ax[3].scatter(i, roughness[i])

    b1.set_data(np.arange(0, i), mse[:i])
    b2.set_text("Sum (C): {}".format(np.round(np.sum(mse[:i]), 2)))
    b3.set_data(np.arange(0, i), mse_2[:i])
    b4.set_text("Sum (A): {}".format(np.round(np.sum(mse_2[:i]), 2)))
    c1.set_data(np.arange(0, i), distance_from_max[:i])
    c2.set_text("Sum (C): {}".format(np.sum(distance_from_max[:i])))
    c3.set_data(np.arange(0, i), distance_from_max_2[:i])
    c4.set_text("Sum (A): {}".format(np.sum(distance_from_max_2[:i])))
    d1.set_data(np.arange(0, i), roughness[:i])
    d2.set_text("Sum (C): {}".format(np.sum(roughness[:i])))
    d3.set_data(np.arange(0, i), roughness_2[:i])
    d4.set_text("Sum (A): {}".format(np.sum(roughness_2[:i])))
    e1.set_data(np.arange(0, i), theta[:i])
    e2.set_text("{}".format(theta[i]))

    return a1, a2, a3, a4, a5, a6, b1, c1, d1, e1

animation = ani.FuncAnimation(fig, animate, len(data), repeat = False)

#animation.save(reconstructions_folder + "animated_hd.gif")

plt.show()

print(np.mean(data))

print(np.mean((data - reconstructions) ** 2))

rat_coordinates = np.load(data_folder + 'filtered_body_poses.npy')[:290]

pointwise_error = np.mean((data - reconstructions) ** 2, axis = 1)

print(pointwise_error.shape)

print(rat_coordinates)

fig, ax = plt.subplots(1,1)

print(rat_coordinates[0, 1])
print(rat_coordinates[0, 2])

#rat_coordinates_time = np.interp(np.arange(0, len(rat_coordinates), 0.1), np.arange(len(rat_coordinates)), rat_coordinates[:, 0])
#rat_coordinates_x = np.interp(np.arange(0, len(rat_coordinates), 0.1), np.arange(len(rat_coordinates)), rat_coordinates[:, 1])
#rat_coordinates_y = np.interp(np.arange(0, len(rat_coordinates), 0.1), np.arange(len(rat_coordinates)), rat_coordinates[:, 2])
#rat_coordinates_theta = np.interp(np.arange(0, len(rat_coordinates), 0.1), np.arange(len(rat_coordinates)), rat_coordinates[:, 3])

#rat_coordinates = np.array([rat_coordinates_time, rat_coordinates_x, rat_coordinates_y, rat_coordinates_theta])

a1 = ax.quiver(rat_coordinates[0, 1], rat_coordinates[0, 2], np.cos(rat_coordinates[0, 3]), np.sin(rat_coordinates[0, 3]), pointwise_error[0] * 255,  label = 'Position', angles = 'xy')
ax.set_title("Train: Centre Rotation")
ax.set_xlim(-3.3, 3.3)
ax.set_ylim(-3.3, 3.3)

def animate(i):

    a1.set_UVC(np.cos(rat_coordinates[i, 3]), np.sin(rat_coordinates[i, 3]), C = pointwise_error[i] * 255)
    a1.set_offsets(np.array([rat_coordinates[i,1], rat_coordinates[i,2]]))
    plt.legend()

    return a1,

animation = ani.FuncAnimation(fig, animate, len(rat_coordinates), repeat = True)

#animation.save(reconstructions_folder + "animated_quiver.gif")

plt.show()
