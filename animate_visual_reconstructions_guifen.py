#Still TODO for Friday; load visual reconstructions for testsets from any/all of testsets 1,2,3,5,9,13,15 and 19, view pose, actual input and reconstruction
#Possibly also add a metric, some kind of Wassenstein metric?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

testset = 9

working_directory = "C:\\Users\\Tom\\Downloads\\HBP\\multimodalplacerecognition_datasets\\whiskeye_guifen_datasets\\real_data_6005_timestamped\\"

fig = plt.figure()
#fig.subplots_adjust(wspace=1.5)
ax1 = fig.add_subplot(3,1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Visual Input")
ax2 = fig.add_subplot(3,1,2, projection = 'polar')
#ax2.set_xticks(np.pi/180. * np.linspace(0,  360, 4, endpoint=False))
ax2.set_yticks([])
ax2.set_theta_offset(np.pi/2)
ax2.set_title("Head Direction", y = 1.25)
ax3 = fig.add_subplot(3,1,3)
ax3.set_yticks([])
ax3.set_xticks([])
ax3.set_title("Visual Reconstruction")

fig.tight_layout()

images = np.load(working_directory + "images.npy").astype(dtype = 'float32')[3000:3050]

poses = np.load(working_directory + "filtered_body_poses.npy").astype(dtype = 'float32')[3000:3050]

working_directory = "C:/Users/Tom/Downloads/HBP/representations/NRP/whiskeye_guifen_real_data_6005_timestamped_Gc5/all/"

reconstructions = np.load(working_directory + "reconstructions_visual.npy").astype(dtype = 'float32')

print(reconstructions.shape)

theta_angles = poses[:,3]

polar_points = []

for angle in theta_angles:
    theta = ax2.plot(angle)
    polar_points.append([theta])

theta_iterator = iter(polar_points)

image_generator = iter(images)

scene = ax1.imshow(images[0], animated = True)

theta_marker, = ax2.plot([theta_angles[0], theta_angles[0]],[0,50], color = 'b', linewidth = 1)

reconstructed_scene = ax3.imshow(reconstructions[0, :, :, :], animated = True)

def animate_both(i):
    scene.set_data(images[i])
    theta_marker.set_data([theta_angles[i], theta_angles[i]],[0,50])
    reconstructed_scene.set_data(reconstructions[i, :, :, :])
    return theta_marker, scene, reconstructed_scene

animated_theta = animation.FuncAnimation(fig, animate_both, interval=1000, blit=True, repeat_delay=10000)

plt.show()