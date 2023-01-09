import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glob import glob
from natsort import natsorted
import os

working_directory = "C:/Users/Thomas/Downloads/HBP/model_checkpoints/landmarks_vh/snow_husky/reconstructions/msi/"

samples_to_display = [0,100,200,300,400]

data = None

minibatch_sz = 10
n_sample = 1000
epochs = 200

cell_width = 63

batches_per_epoch = n_sample // minibatch_sz

if not os.path.exists(working_directory + 'reconstructions.npy'):

    reconstructions = natsorted(glob(working_directory + '*.npy'))

    data = []
    epoch = []
    count = 0

    for reconstruction in reconstructions:

        epoch.append(np.load(reconstruction))

        count += 1

        if count == batches_per_epoch:

            data.append(epoch)

            epoch = []

            count = 0

    data = np.array(data)
    
    np.save(working_directory + 'reconstructions.npy', data)

else:

    data = np.load(working_directory + 'reconstructions.npy')

# Data is gathered as (epoch, minibatch group, minibatch, reconstruction)
# For simplicity, this is reshaped into the form (epoch, sample, reconstruction)

data = np.reshape(data, (200, 1000, cell_width))

ground_truth_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/snow_husky_trainingset/'

ground_truth = np.load(ground_truth_path + 'networkOutput_gaussianised.npy')[0:n_sample]#'networkOutput.npy').T[0:n_sample]

#print(ground_truth.shape)

# for sample in samples_to_display:

#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)

#     subset = []

#     for epoch in data: # For each epoch

#         subset.append(epoch[sample])

#     predicted, = ax1.plot(range(len(subset[0])), subset[0])
#     actual, = ax1.plot(range(len(subset[0])), ground_truth[0])

#     def animate_reconstruction(i):
#         predicted.set_data(range(len(subset[i])), subset[i])
#         actual.set_data(range(len(subset[i])), ground_truth[i])
#         ax1.set_ylim(min(subset[i] * 1.2), max(subset[i] * 1.2))
#         return predicted, actual,

#     animated = animation.FuncAnimation(fig, animate_reconstruction, interval=100, blit=True, repeat_delay=10000)

#     plt.show()

data = data.reshape(n_sample * epochs, cell_width)

#print(data.shape)

#print(ground_truth.shape)

ground_truth = np.expand_dims(ground_truth, 0)
ground_truth = np.repeat(ground_truth, epochs, axis = 0)
ground_truth = np.reshape(ground_truth, (n_sample * epochs, cell_width))

#print(ground_truth.shape)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

predicted, = ax1.plot(range(len(data[0])), data[0], label = "Reconstruction")
actual, = ax1.plot(range(len(ground_truth[0])), ground_truth[0], label = "Ground Truth")
epoch_text = ax1.text(52,0.98, 'Epoch:{}'.format(1))
sample_text = ax1.text(51.9,0.90, 'Sample:{}'.format(1))
ax1.set_xlabel("Head Direction Cell")
ax1.set_ylabel("Reconstruction Activity")
ax1.set_title("Training History")
ax1.legend()

def animate_reconstruction(i):
    predicted.set_data(range(len(data[i])), data[i])
    actual.set_data(range(len(ground_truth[i])), ground_truth[i])
    #ax1.set_ylim(min(ground_truth[i] / 1.2), max(ground_truth[i] * 1.2))
    if i % 1000 == 0:
        epoch_text.set_text('Epoch:{}'.format(i//1000))
    sample_text.set_text('Sample:{}'.format(i % 1000))
    return predicted, actual, epoch_text, sample_text,

animated = animation.FuncAnimation(fig, animate_reconstruction, interval=1, blit=True, repeat_delay=10000)

plt.show()