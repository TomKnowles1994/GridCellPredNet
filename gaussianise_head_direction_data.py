import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sharpness = 20
plot = True

cell_format = False

rescale = True

distribution = "Gaussian"

def preprocess_pose_data(pose_data):
    scaler = MinMaxScaler(copy=False)
    scaler.fit(pose_data)
    scaler.transform(pose_data)

    return pose_data

def gaussianise(data_path, sharpness, plot = False, cell_format = False, dataset_type = 'other'):

    N = 180

    if cell_format is True:

        poses = np.load(data_path + '/networkOutput.npy')

        if dataset is not None:

            print("Testset {} starting shape: {}".format(dataset, poses.shape))

        else:

            print("Trainingset starting shape: {}".format(poses.shape))

    if cell_format is False:

        #poses = np.load(data_path + '/training_set_ideo_estimate_byFrameTime.npy')[:,:391]
        #poses = np.load(data_path + '/poses.npy')
        if dataset_type == 'rotation': # Rotation datasets take some time to relocate to their true position, so we trim from the start
            poses = np.load(data_path + '/filtered_body_poses.npy')[15:,3]
        elif dataset_type == 'translation': # Translation datasets additionally cause a big ruckus when they bang into walls, so we trim from the end too
            poses = np.load(data_path + '/filtered_body_poses.npy')[15:-15,3]
        else:
            poses = np.load(data_path + '/filtered_body_poses.npy')[:,3]

        if dataset is not None:

            print("Testset {} starting shape: {}".format(dataset, poses.shape))

        else:

            print("Trainingset starting shape: {}".format(poses.shape))

        if len(poses.shape) > 1:

            if poses.shape[1] == 10: # If pose contains full quaternion (such as gazebo_pose.npy)

                theta = poses[:,9]
            
            elif poses.shape[1] > 3 and poses.shape[1] < 7: # If pose contains index and timestamp
                
                theta = poses[:,4]

            elif poses.shape[1] == 3: # If pose does not

                theta = poses[:,2]

            elif poses.shape[0] == 2: # If in idiothetic estimate form

                poses = poses.T
                theta = poses[:,1]

        else:

            theta = poses

        N = 180 #Used to be 63
        
        angle_per_cell = (2*np.pi)/N
        in_cell = (theta//angle_per_cell) + np.ceil(N/2)
        
        networkOutput = np.zeros((N,len(in_cell)))
        activeCells = np.zeros((5,len(in_cell)))
        activeTime = np.zeros((5,len(in_cell)))
        
        for i,L in enumerate(in_cell):
            index = np.arange(-2,3) + np.int(L)
            index[index<0] = index[index<0]+N
            index[index>=N] = index[index>=N]-N 
            activeCells[:,i] = index
            activeTime[:,i] = i
            activity = [1,1,2,1,1]
            #activity = [0.5,0.5,1,0.5,0.5]
            #activity = [0,0,1,0,0]
            #activity = [0.5,0.5,1,1,2,1,1,0.5,0.5]
            networkOutput[index,i] = activity

        #np.save(data_path + '/networkOutput_ideo_single_rotation.npy', networkOutput)
        np.save(data_path + '/networkOutput.npy', networkOutput)
            
        poses = networkOutput.T

    gaussian_width = poses.shape[1]

    gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2)) # Used to be up to (gaussian_width//2)+1 (adjusts for odd numbers)

    gaussian_block = np.resize(gaussian_range, new_shape = (poses.shape))

    max_locations = np.argmax(poses, axis = 1)

    if dataset is not None:

        print("Testset {} end shape: {}".format(dataset, max_locations.shape))

    else:

        print("Trainingset end shape: {}".format(max_locations.shape))

    if distribution is "Gaussian":

        pose_gaussians = norm(0, gaussian_width//sharpness)

    if distribution is "Laplace":

        pose_gaussians = laplace(0, gaussian_width//sharpness)

    zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

    shifted_gaussians = np.empty_like(poses)

    shifted = np.roll(zeroed_gaussians[0], max_locations[0]-(N//2)) # Used to be hard-coded to -31 i.e. -63//2

    print(max_locations)

    for index in range(len(max_locations)):
        shifted = np.roll(zeroed_gaussians[index,:], max_locations[index]-(N//2))
        shifted_gaussians[index,:] = shifted

    if rescale:

        scaling_factor = 1/shifted_gaussians.max()
        shifted_gaussians = shifted_gaussians * scaling_factor#preprocess_pose_data(shifted_gaussians)

    if plot:

        fig, ax = plt.subplots(1, 1)

        for i in range(0, 60, 10):
            ax.plot(shifted_gaussians[i,:])
            if not rescale:
                ax.vlines(max_locations[i], 0., 0.07, color = 'grey')
            if rescale:
                ax.vlines(max_locations[i], 0., 1.1, color = 'grey')
            plt.xlabel("Head Direction")
            plt.ylabel("Pseudo-probability")
            plt.title("Ground Truth Data")

        plt.show()

    #np.save(data_path + '/networkOutput_gaussianised_ideo_single_rotation.npy', shifted_gaussians)
    np.save(data_path + '/networkOutput_gaussianised', shifted_gaussians)

# dataset = None

# gaussianise(data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped', 
#              sharpness = sharpness, 
#              plot = plot, 
#              cell_format = cell_format)

for dataset in ("training_data_centre", "training_data_top_left", "training_data_top_right", "training_data_bottom_left", 
                "training_data_bottom_right", "training_data_centre_5x"):

    gaussianise(data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_gazebo_3_3000/{}'.format(dataset), 
                sharpness = sharpness, 
                plot = plot, 
                cell_format = cell_format,
                dataset_type = 'rotation')

for dataset in ("training_data_bl_tr", "training_data_tr_bl", "training_data_br_tl", "training_data_tl_br", 
                "training_data_lm_rm", "training_data_rm_lm", "training_data_bm_tm", "training_data_tm_bm"):

    gaussianise(data_path = 'C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/training_data/training_data_translation_sets_3000/{}'.format(dataset), 
                sharpness = sharpness, 
                plot = plot, 
                cell_format = cell_format,
                dataset_type = 'translation')

#dataset = 1
#if False:
#for dataset in (1,6,10,11,12):#range(1,21):#(6,10,11,12):
#for dataset in ("rotating_distal", "rotating_proximal", "circling_distal", "circling_proximal", "random_walk_distal", "random_walk_proximal", "cogarch"):

    #gaussianise(data_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_rotating_testset{}_30Hz_husky_speed'.format(dataset, dataset), 
    #            sharpness = sharpness, 
    #            plot = plot, 
    #            cell_format = cell_format)
    #gaussianise(data_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_{}'.format(dataset), 
    #            sharpness = sharpness, 
    #            plot = plot, 
    #            cell_format = cell_format)