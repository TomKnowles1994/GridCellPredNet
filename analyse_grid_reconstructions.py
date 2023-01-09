import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data_folder = "C:\\Users\\Tom\\Downloads\\HBP\\multimodalplacerecognition_datasets\\whiskeye_guifen_datasets\\real_data_6005_timestamped\\"

data = np.load(data_folder + 'PredNet_grid_data.npy')[3000:3050] # 2500 Test Samples

#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_6005_timestamped_reconstructions_no_visual\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_representations_no_grid\\"
reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\whiskeye_guifen_real_data_6005_timestamped_Gc5_VisHd\\no_grid\\"

reconstructions = np.load(reconstructions_folder + 'reconstructions_grid_cells.npy')

#reconstructions = reconstructions.reshape(reconstructions.shape[0], 5, 1000)

print(data.shape)
print(reconstructions.shape)

fig, ax = plt.subplots(5, 1, sharex = True)

ax = ax.flatten()

for n, i in enumerate(range(0, 50, 10)):
    for j in range(data.shape[1]):
        if j == 0:
            ax[n].plot(np.arange(1000), data[i,j,:])
            ax[n].plot(np.arange(1000), reconstructions[i,j,:])
        else:
            ax[n].plot(np.arange(1000), data[i,j,:], alpha = 0.3)
            ax[n].plot(np.arange(1000), reconstructions[i,j,:], alpha = 0.3)
            
plt.show()

print(np.mean(data))