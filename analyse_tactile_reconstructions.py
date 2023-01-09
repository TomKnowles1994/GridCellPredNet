import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data_folder = "C:\\Users\\Tom\\Downloads\\HBP\\multimodalplacerecognition_datasets\\whiskeye_guifen_datasets\\real_data_6005_timestamped\\"

data = np.loadtxt(data_folder + 'whisker_data.csv', delimiter = ',', skiprows = 1)[3000:5500] # 2500 Test Samples

#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_6005_timestamped_reconstructions_no_visual\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_representations_no_grid\\"
reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\whiskeye_guifen_real_data_6005_timestamped\\no_grid\\"

reconstructions = np.load(reconstructions_folder + 'reconstructions_tactile.npy')

print(data.shape)
print(reconstructions.shape)

non_zero_rows = np.unique(np.nonzero(data)[0])

# for i in range(0, 2500, 125):

#     plt.plot(np.arange(data.shape[1]), data[i,:])
#     plt.plot(np.arange(data.shape[1]), reconstructions[i,:])
#     plt.show()

for i in non_zero_rows[0:5]:

    plt.plot(np.arange(data.shape[1]), data[i,:])
    plt.plot(np.arange(data.shape[1]), reconstructions[i,:])
    plt.show()

print(np.mean(data))