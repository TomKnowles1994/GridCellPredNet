import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace

gaussian_method = 'new'
show_sample_output = True
merge_scales_before_export = False
output_filetype = 'numpy'

# number of neurons in each ring triplet for each scale
# n = [300 420 590 830 1170]
n = np.array([30, 40, 60, 80, 120])
#n = np.array([10, 14, 20, 28, 40])
num_scales=len(n)
# starting position of "bump" in neural index for each ring (same in all
# scales)
startbump = np.array([0, 0, 0])
# velocity gain to set grid spacing
VelGain = 15
# number of ring attractors in each scale
num_rings = 3
# angle of grid #axis from 0 degrees (+/-)
Gridangle = 60
Axis2 = np.radians(Gridangle)    # angle of second non-rectangular #axis
Axis3 = np.radians(-Gridangle)   # angle of third non-rectangular #axis

# receptive field for each RP
spread = np.array([3, 4, 6, 8, 12])

# number of Ring Phase cells per ring (depends on spread)
num_RP = np.zeros(shape=(num_scales))

# number of grid cells representing each scale
num_grid_cells = np.zeros(shape=(num_scales))

num_RP = np.divide(n, spread).astype('int')
num_grid_cells = num_RP**num_rings

grid_cells = {'Scales': None, 'State': None, 'Conns': None}

grid_cells['Scales'] = num_grid_cells

## Make spiral trajectory with constant speed
V = 5
dr = 10
ts =np.linspace(0,3600,18000)

ph =np.sqrt(((V * (4*np.pi) * ts) / dr))
ra =np.sqrt(((V * dr * ts) / np.pi))

pos_x = ra*np.cos(ph)
pos_y = ra*np.sin(ph)


#data = sio.loadmat("Sargolini.circle_11025-14050501.mat")

#pos_x = data['pos_x'].squeeze()
#pos_y = data['pos_y'].squeeze()

data_filepath = "C:/Users/Tom/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_guifen_datasets/real_data_6005_timestamped/"

#data = np.genfromtxt(data_filepath + "body_poses.csv", dtype = np.float64, delimiter = ",", skip_header = 1)
data = np.load(data_filepath + 'filtered_body_poses.npy')

#pos_x = data[:,0]# - np.min(data[:,0])
#pos_y = data[:,1]# - np.min(data[:,1])

pos_x = data[:,1]# - np.min(data[:,0])
pos_y = data[:,2]# - np.min(data[:,1])

N = len(pos_x)

mag = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)
ang = np.arctan2(np.diff(pos_y), np.diff(pos_x))
#print(ang)
N = len(mag)

#check integration
# [pos_x_prime pos_y_prime] = pol2cart(ang,mag)
# pos_x_prime = cumsum(pos_x_prime)
# pos_y_prime = cumsum(pos_y_prime)
# 
# pos_x_prime = pos_x_prime+pos_x(1)
# pos_y_prime = pos_y_prime+pos_y(1)
# hold on
# plot(pos_x_prime, pos_y_prime,'r')

# split veocity into 3 components
F       = np.zeros(shape=(num_scales,num_rings,N))    # current injection into each ring
Rings   = np.zeros(shape=(num_scales,num_rings,N))    # ring activity

grid_cells['State'] = [None for i in range(num_scales)]

for i in range(num_scales):
    grid_cells['State'][i] = np.zeros(shape=(grid_cells['Scales'][i].astype('int'),N))    

#grid_cells = np.zeros(num_scales,num_grid_cells,N)
# starting position of "bump" in neural index 
Rings[:,0,0] = startbump[0]
Rings[:,1,0] = startbump[1]
Rings[:,2,0] = startbump[2]

# run the rings
for i in range(N):
    #angle = wrapToPi(np.radians[i])      # angle of motion vector    
    angle = ang[i]
    Mag = mag[i]
    for k in range(num_scales):
        # Assign credit from velocity command to inject into each ring attractor
        F[k,0,i] = VelGain*(Mag*np.cos(angle))
        F[k,1,i] = VelGain*(Mag*np.cos(Axis2-angle))
        F[k,2,i] = VelGain*(Mag*np.cos(Axis3-angle))

        # inject current into each ring (integrate)
        for j in range(num_rings):
            if i==0:
                Rings[k,j,i] = Rings[k,j,i]+F[k,j,i]
            else:
                Rings[k,j,i] = Rings[k,j,i-1]+F[k,j,i]
                
            # wrap the size of each ring to n neurons
            if Rings[k,j,i] > n[k]:
                Rings[k,j,i] = Rings[k,j,i]-n[k]
            elif Rings[k,j,i] < 0:
                Rings[k,j,i] = Rings[k,j,i]+n[k]

# # Assign credit from velocity command to inject into each ring attractor
# F[:,0,:] = VelGain*(mag*np.cos(ang))
# F[:,1,:] = VelGain*(mag*np.cos(Axis2-ang))
# F[:,2,:] = VelGain*(mag*np.cos(Axis3-ang))

# for i in range(N):
#     # inject current into each ring (integrate)
#     if i==0:
#         Rings[:,:,i] = Rings[:,:,i]+F[:,:,i]
#     else:
#         Rings[:,:,i] = Rings[:,:,i-1]+F[:,:,i]
    
#     # Get more copies of n (per ring neuron count) to do all scales and rings at once
#     n_r = np.repeat(n, num_rings).reshape(num_scales, num_rings)

#     # wrap the size of each ring to n neurons
#     Rings[:,:,i][Rings[:,:,i] > n_r] = Rings[:,:,i][Rings[:,:,i] > n_r] - n_r[Rings[:,:,i] > n_r]
#     Rings[:,:,i][Rings[:,:,i] < n_r] = Rings[:,:,i][Rings[:,:,i] < n_r] + n_r[Rings[:,:,i] < n_r]
                        
#Resultant =np.sqrt(F(1,:).^2 + F(2,:).^2 + F(3,:).^2) # check this is balanced
#convert to integer form
Rings = Rings.astype('int')

RP_cells = {'State': None, 'Conns': None}

RP_cells['Conns'] = [None for i in range(num_scales)]

## set connection matrix for RP cells to rings

for i in range(num_scales):
    RP_cells['Conns'][i] = np.zeros(shape=(num_rings,num_RP[i]))

for k in range(num_scales):  # for each scale represented
    for x in range(num_rings): # for each ring in each scale (nominally 3)
        for y in range(num_RP[k]): # for each RP cell in each scale for each ring
            RP_cells['Conns'][k][x,y]=y*spread[k] + 1 # pointer to start index in the ring

RP_cells['State'] = [None for i in range(num_scales)]

## run the RP cells
for i in range(num_scales):
    RP_cells['State'][i] = np.zeros(shape=(num_rings,num_RP[i],N))    

for k in range(num_scales):
    for i in range(N):
        for r in range(num_rings):
            for p in range(num_RP[k]):
                if Rings[k,r,i] >= RP_cells['Conns'][k][r,p] and Rings[k,r,i] < RP_cells['Conns'][k][r,p]+spread[k]:
                    RP_cells['State'][k][r,p,i] = 1
    #print(RP_cells['State'][k])
grid_cells['Conns'] = [None for i in range(num_scales)]

##
# set connection matrix for grid cells to RP cells (regular distribution)
for i in range(num_scales):
    grid_cells['Conns'][i] = np.zeros(shape=(num_grid_cells[i],num_rings))

count = 0
for k in range(num_scales):     # cycle through each scale
    for j in range(num_RP[k]):
        for i in range(num_RP[k]):   # different number of RP in each scale
            for m in range(num_RP[k]):
                grid_cells['Conns'][k][count,:] = [j, i, m]
                count=count+1
    count = 0
    
## run the grid cells

for i in range(N): # for each time step in the set
    for k in range(num_scales): 
        for g in range(grid_cells['Scales'][k]): # for each grid cell
            if RP_cells['State'][k][0,int(grid_cells['Conns'][k][g,0]),i] == 1 and RP_cells['State'][k][1,int(grid_cells['Conns'][k][g,1]),i] == 1 and RP_cells['State'][k][2,int(grid_cells['Conns'][k][g,2]),i] == 1:
                grid_cells['State'][k][g,i] = 1
                #print("True for scale {}".format(n[k]))
                            
 ## plot stripe cells (RP cells)

#fig, #ax = plt.subplots(1,num_scales, #figsize = (15, 3))

for k in range(num_scales):

    #subplot(2,3,k)
    I=np.argwhere(RP_cells['State'][k][0,0,:])
    #ax[k].scatter(pos_x[I],pos_y[I], c = 'b')
    I=np.argwhere(RP_cells['State'][k][1,0,:])
    #ax[k].scatter(pos_x[I],pos_y[I], c = 'r')
    I=np.argwhere(RP_cells['State'][k][2,0,:])
    #ax[k].scatter(pos_x[I],pos_y[I],c = 'g')

#plt.show()

## plot most active grid cell in each scale

#fig, #ax = plt.subplots(1,num_scales, #figsize = (15, 3))

for k in range(num_scales):
    #ax[k].plot(pos_x,pos_y,'b')
    max_len = 0
    #subplot(2,3,k)
    for i in range(grid_cells['Scales'][k]):
        I=np.argwhere(grid_cells['State'][k][i,:])
        #print(I)
        if I.size > 0:
            if max_len < len(I):
                #plot(pos_x(I),pos_y(I),'color',c{1},'marker','*')
                #ax[k].plot(pos_x[I],pos_y[I],'*', c = 'red')
                max_len = len(I)

#plt.show()

# Plot all grid cells per scale as a heatmap

#fig, #ax = plt.subplots(1,num_scales, #figsize = (15, 3))

for k in range(num_scales):
    ##ax[k].plot(pos_x,pos_y,'b')
    max_len = 0
    #subplot(2,3,k)
    I = np.argwhere(grid_cells['State'][k][:,:])
    #ax[k].hexbin(pos_x[I],pos_y[I], cmap = 'Reds', gridsize = (15,15))

#plt.show()

## analyis
grid_codes= np.zeros(shape=(num_scales,N))
for i in range(N):
    for k in range(num_scales):
        I = np.argwhere(grid_cells['State'][k][:,i])
    if I.size == 0: # possibly should be: if I.shape[0] == 0
        grid_codes[k,i] = 0
    else:
        grid_codes[k,i] = I

active_gridcells = {'GT Pose': None, 'Count': None, 'Est Pose': None}

## looking for coincident grid cells
active_gridcells['GT Pose']  = np.zeros(shape = (2,N))      # copy of actual location for this data point
active_gridcells['Count']    = np.zeros(shape = (N))        # how many grid cell codes active 
active_gridcells['Est Pose'] = np.zeros(shape = (2,10,N))      # list of places that are represented by the same grid code

size_of_test = len(mag)#2000

for i in range(size_of_test): # cycle through each data point in the set
    active_gridcells['GT Pose'][:,i] = [pos_x[i], pos_y[i]]
    sample = grid_codes[:,i] # read grid code associated with that sample
    if ~np.all(sample): # this was [0 0 0 0], shouldn't that be [0 0 0 0 0] (list of zeroes of length num_scales?)
        active_gridcells['Count'][i] = 0 # no grid codes representing this data point
    else:
        count = 0
        for j in range(len(grid_codes)): # cycle through grid codes for each data point
            if sample == grid_codes[:,j]: # compare each code with the current cycle
                count = count + 1
                active_gridcells['Est Pose'][:,count,i] = [pos_x(j), pos_y(j)] 
                active_gridcells['Count'][i] = count
##
##figure(5)

#fig, #ax = plt.subplots(1,1)

##ax.plot(pos_x[1:i],pos_y[1:i], c = 'b')

#for i in range(size_of_test):
#    for j in range(int(active_gridcells['Count'][i])):
        ##ax.plot(active_gridcells['Est Pose'][0,j,i],active_gridcells['Est Pose'][1,j,i], marker = '*', c = 'red')
    
    ##ax.plot(pos_x[:i],pos_y[:i], c = 'b')
    #ax.set_xlim(-100, 100)
    #ax.set_ylim(-100, 100)
    #ax.set_title("{}".format(i))
    # might need to use the animation API for this

## create a 1D Gaussian to insert into grid cell space 
#gaus = @(x,mu,sig,amp,vo)amp*exp(-(((x-mu).^2)/(2*sig.^2)))+vo

def gaus(x,mu,sig,amp,vo):

    return amp*np.exp(-(((x-mu)**2)/(2*sig**2)))+vo

x =np.linspace(0,200,400)
mu = 100
sig = 10
amp = 1
vo = 0
y = gaus(x,mu,sig,amp,vo)

#fig, #ax = plt.subplots(1,1)

##ax.plot(x, y, 'b-', 'LineWidth',3)

#plt.show()

if gaussian_method == 'old':

    ## insert Gaussians into grid cell activities
    for i in range(N):
        for k in range(num_scales):
            I = np.argwhere(grid_cells['State'][k][:,i])
            if I.size > 0:
                grid_cells['State'][k][I,i] = y[mu*2]
                for j in range(mu*2):
                    if I-j <= 0:
                        index = I-j + num_grid_cells
                    else:
                        index = I-j
                        grid_cells['State'][k][index,i] = y[(mu*2)-j]
                        for j in range(mu*2):
                            if I+j >num_grid_cells[k]:
                                index = I+j - num_grid_cells[k]
                            else:
                                index = I+j-1
                                grid_cells['State'][k][index,i] = y[j+(mu*2)]
                            ## construct prednet input set
        print("Processed Sample {}/{}".format(i+1, N), end = '\r')
    print("\n")

    PredNetInput=np.zeros(shape=(N,num_scales,1000))
    for i in range(N):
        PredNetInput[i,0,:] = grid_cells['State'][0][:,i].T
        PredNetInput[i,1,:] = grid_cells['State'][1][:,i].T
        PredNetInput[i,2,:] = grid_cells['State'][2][:,i].T
        PredNetInput[i,3,:] = grid_cells['State'][3][:,i].T
        PredNetInput[i,4,:] = grid_cells['State'][4][:,i].T
        
        print("Saving Sample {}/{}".format(i+1, N), end = '\r')
    print("\n")

elif gaussian_method == 'new':

    PredNetInput=np.zeros(shape=(N,num_scales,1000))

    # For each scale
    for k in range(num_scales):

        # Create a block of indices to store N gaussians, all centered on 0
        gaussian_width = len(grid_cells['State'][k][:,0])
        gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2))
        gaussian_block = np.resize(gaussian_range, new_shape = (N, gaussian_width))

        # Find where the max (active grid cell) is for each sample
        max_locations = np.argmax(grid_cells['State'][k], axis = 0)

        # Create a function for a 0-mean Gaussian with the desired sigma
        pose_gaussians = norm(0, sig)

        # Apply this function onto the block of indices, giving N Gaussians all with mean 0
        zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

        # Preallocate for final Gaussians
        shifted_gaussians = np.empty_like(zeroed_gaussians)

        # Move each Gaussian to its proper position, centred over the active grid cell
        for index in range(len(max_locations)):
            shifted_gaussians[index, :] = np.roll(zeroed_gaussians[index, :], max_locations[index]-(N//2))

        # Rescale so that the Gaussians are in range 0-1
        scaling_factor = 1/np.max(shifted_gaussians, axis = 1)
        shifted_gaussians = shifted_gaussians.T * scaling_factor

        # Assign to the output numpy array
        PredNetInput[:,k,:] = shifted_gaussians.T
        
        print("Saving Scale {}/{}".format(k+1, num_scales), end = '\r')
    print("\n")

if show_sample_output:

    for k in range(num_scales):

        plt.plot(PredNetInput[0, k, :])
        plt.plot(PredNetInput[100, k, :])
        plt.plot(PredNetInput[200, k, :])
    plt.show()

if output_filetype == 'numpy':

    if merge_scales_before_export:

        PredNetInput = PredNetInput.reshape(PredNetInput.shape[0], PredNetInput.shape[1] * PredNetInput.shape[2])

    np.save(data_filepath + "PredNet_grid_data.npy", PredNetInput)

    print(".npy saved")

elif output_filetype == 'mat':

    if merge_scales_before_export:

        PredNetInput = PredNetInput.reshape(PredNetInput.shape[0], PredNetInput.shape[1] * PredNetInput.shape[2])

        sio.savemat(data_filepath + "PredNet_grid_data.mat", {'grid_cell_activity': PredNetInput})

    elif not merge_scales_before_export:

        sio.savemat(data_filepath + "PredNet_grid_data.mat", {'grid_cell_activity': {   'scale_1': PredNetInput[:,0,:],
                                                                                        'scale_2': PredNetInput[:,1,:],
                                                                                        'scale_3': PredNetInput[:,2,:],
                                                                                        'scale_4': PredNetInput[:,3,:],
                                                                                        'scale_5': PredNetInput[:,4,:] }})

    print(".mat saved")