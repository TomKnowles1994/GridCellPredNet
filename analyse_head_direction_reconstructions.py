import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data_folder = "C:\\Users\\Tom\\Downloads\\HBP\\multimodalplacerecognition_datasets\\whiskeye_guifen_datasets\\real_data_6005_timestamped\\"

data = np.load(data_folder + 'networkOutput_gaussianised.npy')[3000:3050] # 2500 Test Samples

#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_6005_timestamped_reconstructions_no_visual\\"
#reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\guifen_representations_no_grid\\"
reconstructions_folder = "C:\\Users\\Tom\\Downloads\\HBP\\representations\\NRP\\whiskeye_guifen_real_data_6005_timestamped_Gc5_VisHd\\no_hd\\"

reconstructions = np.load(reconstructions_folder + 'reconstructions_hd_cells.npy')

print(data.shape)
print(reconstructions.shape)

non_zero_rows = np.unique(np.nonzero(data)[0])

for i in range(0, 50, 10):

    plt.plot(np.arange(data.shape[1]), data[i,:])
    plt.plot(np.arange(data.shape[1]), reconstructions[i,:])
    plt.show()

print(np.mean(data))

print(np.mean((data - reconstructions) ** 2))