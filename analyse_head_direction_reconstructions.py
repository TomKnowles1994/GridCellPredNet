import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.animation as ani

#data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

#data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/training_data_centre/"
data_folder = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

#data = np.load(data_folder + 'networkOutput_gaussianised.npy')[3000:3050] # 2500 Test Samples

data = np.load(data_folder + 'networkOutput_gaussianised.npy')[:490]

#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_6005_timestamped_reconstructions_no_visual\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_representations_no_grid\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\whiskeye_guifen_real_data_6005_timestamped_Gc5_VisHd\\no_hd\\"
#reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_centre/training_data_centre/visual/"
#reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_rat_centre/real_data_6005_timestamped/visual/"
reconstructions_folder = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_vis_hd_rat_all/real_data_6005_timestamped/visual/"

#reconstructions = np.load(reconstructions_folder + 'reconstructions_hd_cells.npy')
reconstructions = np.load(reconstructions_folder + 'reconstructions_head_direction.npy')

print(data.shape)
print(reconstructions.shape)

non_zero_rows = np.unique(np.nonzero(data)[0])

# for i in range(0, 100, 10):

#     plt.plot(np.arange(data.shape[1]), data[i,:])
#     plt.plot(np.arange(data.shape[1]), reconstructions[i,:])
#     plt.show()

fig, ax = plt.subplots(1,1)

a1 = ax.plot(np.arange(data.shape[1]), data[0,:], label = 'Test Data')[0]
a2 = ax.plot(np.arange(data.shape[1]), reconstructions[0,:], label = 'Reconstruction')[0]
a3 = ax.axvline(np.argmax(reconstructions[0,:]), 0, 1, color = 'green', label = 'Reconstruction Peak')
a4 = ax.text(0, 0.02, "{}/{}".format(0, len(data)))
#ax.set_ylim(0, 1)
ax.set_title("Train: 5 Perspective Rotation; Test: Real Rat")

def animate(i):

    a1.set_data(np.arange(data.shape[1]), data[i,:])
    a2.set_data(np.arange(data.shape[1]), reconstructions[i,:])
    a3.set_data(np.argmax(reconstructions[i,:]), [0,1])
    a4.set_text("{}/{}".format(i, len(data)))
    plt.legend()

    return a1, a2, a3, a4

animation = ani.FuncAnimation(fig, animate, len(data), repeat = True)

animation.save(reconstructions_folder + "animated_hd.gif")

plt.show()

print(np.mean(data))

print(np.mean((data - reconstructions) ** 2))