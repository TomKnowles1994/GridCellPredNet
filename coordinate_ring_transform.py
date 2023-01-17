import numpy as np
import matplotlib.pyplot as plt

data_filepath = "C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

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

    sign_x = np.sign(x)
    sign_y = np.sign(y)

    ring1 = y

    x_component = x * np.sin(np.radians(60))
    y_component = y * np.sin(np.radians(30))

    ring2 = x_component + y_component

    x_component = x * np.sin(np.radians(60+60))
    y_component = y * -np.sin(np.radians(30))

    ring3 = x_component + y_component

    return ring1, ring2, ring3

def ring2cart(ring1, ring2, ring3):

    sign_1 = np.sign(ring1)
    sign_2 = np.sign(ring2)
    sign_3 = np.sign(ring3)

    #x = (np.sqrt(sign_2 * ring2 ** 2 - sign_1 * (ring1 * np.sin(np.radians(30))) ** 2) + np.sqrt(sign_3 * ring3 ** 2 - sign_1 * (ring1 * np.sin(np.radians(30))) ** 2)) / 2
    #y = ring1

    #x = ((sign_2 * ring2 * np.sin(np.radians(60))) + (sign_3 * ring3 * np.sin(np.radians(60)))) / 2
    #y = ring1

    #ring1_component = (sign_1 * ring1 * np.sin(np.radians(30))) * sign_2 * ring2 * np.sin(np.radians(60))
    #ring2_component = sign_2 * ring2 * np.sin(np.radians(60))
    #ring3_component = sign_3 * ring3 * np.sin(np.radians(60))

    #x = ring2_component - ring1_component
    #y = ring1

    ring2_x_component = ring2 / np.sin(np.radians(60))
    ring3_x_component = ring3 / np.sin(np.radians(120))

    x = (ring2_x_component + ring3_x_component) / 2
    y = ring1

    return x, y

# Test points

for i in np.arange(-3, 3.5, 0.5):

    j = 1/i

    ax.plot(i, j, c = 'green', marker = '+', markersize = 30)
    ax.plot(*ring2cart(*cart2ring(i, j)), c = 'orange', marker = '+', markersize = 30)

plt.show()

# Convert datasets

data_folder = "C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/"

data_files = [  "training_data_bl_tr/", "training_data_bm_tm/", "training_data_br_tl/", "training_data_lm_rm/",
                "training_data_rm_lm/", "training_data_tl_br/", "training_data_tm_bm/", "training_data_tr_bl/"]

for data_file in data_files:

    x_y_data = np.loadtxt(data_folder + data_file + "body_poses.csv", delimiter = ',', skiprows = 1)[150:-150,1:3]

    print(data_file)

    print(x_y_data)

    x_y_rings = cart2ring(x_y_data[0], x_y_data[1])

    x_y_cart = ring2cart(x_y_rings[0], x_y_rings[1], x_y_rings[2])

    plt.plot(x_y_cart)

    plt.show()