#!/home/tomknowles/catkin_ws/whiskeye_venv/bin python3

from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from whiskeye_msgs.msg import *

import rospy
import random
import math
import kc_interf
import numpy as np
import scipy as sp
import pandas as pd
import os
import cv2
import time

np.set_printoptions(suppress = True, sign = ' ')

data_folder = '/home/tomknowles/Desktop/'

data_type = 'Train' # 'Train' (synthetic data) or 'Test' (rat data)

training_sets = {   'Centre'        : data_folder + 'rotation_dataset_centre.csv',
                    'Top Left'      : data_folder + 'rotation_dataset_top_left.csv',
                    'Top Right'     : data_folder + 'rotation_dataset_top_right.csv',
                    'Bottom Left'   : data_folder + 'rotation_dataset_bottom_left.csv',
                    'Bottom Right'  : data_folder + 'rotation_dataset_bottom_right.csv',
                    'Translation'   : data_folder + 'translation_dataset.csv'}
                    
training_set = training_sets['Centre']
                    
test_set = data_folder + '2dVR_data.mat'

data_set = 'Real' # 'Real' or 'VR' data

image_channels = 3 # RGB

image_element_size = 1 # bytes

class controller():

    def __init__(self):
    
        ### Settings ###

        self.update_rate = 50.0

        self.fs = 1.0/self.update_rate

        self.whisk_rate = 5

        self.destination = None
        
        self.index = 0
        
        self.image_index = 0
        
        #self.image_timestamp = time.time()
        
        ### 2-unit sliding window ###
        
        self.prior_whisker_theta = 0
        self.current_whisker_theta = 0
        self.peaked = False
        
        ### Data caches ###
        
        self.data_x             = None  # Matlab Field #5,1
        self.data_y             = None  # Matlab Field #5,2
        self.data_body_theta    = None  # Matlab Field #6
        self.data_head_theta    = None  # Matlab Field #7
        self.data_speed         = None  # Matlab Field #8
        self.data_time          = None  # Matlab Field #9
        
        self.neck_angles = np.array([0.0, 0.0, 0.0])
        self.body_pose = np.array([0.0, 0.0, 0.0])
        self.whisker_xy = np.zeros(shape = (2, 24), dtype = np.float64)
        self.image_data = None
        
        ### Folder and file setup
        
        self.output_folder = os.getcwd() + '/output/'
        
        if os.path.exists(self.output_folder):
        
            print("Output folder {}".format(self.output_folder))
        
        elif not os.path.exists(self.output_folder):
            
            try:
            
                os.mkdir(self.output_folder)
            
                print("Output folder created at {}".format(self.output_folder))
            
            except IOError:
            
                pass
        
        self.image_folder = self.output_folder + 'images/'
        
        if os.path.exists(self.image_folder):
        
            print("Saving camera images to {}".format(self.image_folder))
        
        elif not os.path.exists(self.image_folder):
            
            try:
            
                os.mkdir(self.image_folder)
            
                print("Image folder created at {}".format(self.image_folder))
            
            except IOError:
            
                pass
        
        
        ### Get file handles and write column headers ###
            
        try:
            
            self.timestamp_file = open(self.output_folder + 'image_timestamps.csv', 'a', buffering = 1)
            
            if not os.path.getsize(self.output_folder + 'image_timestamps.csv'): # If there's nothing in the file yet
            
                self.timestamp_file.write("{}\n".format("Time"))
                
        except IOError:
        
            print("Error writing to 'image_timestamps.csv'")
            
            raise IOError
            
        try:
        
            self.body_pose_file = open(self.output_folder + 'body_poses.csv', 'a', buffering = 1)
            
            if not os.path.getsize(self.output_folder + 'body_poses.csv'):
        
                self.body_pose_file.write("{},{},{},{}\n".format("Time", "X", "Y", "Theta"))
                
        except IOError:
        
            print("Error writing to 'body_poses.csv'")
            
            raise IOError
        
        try:
        
            self.whisker_file = open(self.output_folder + 'whisker_data.csv', 'a', buffering = 1)
            
            if not os.path.getsize(self.output_folder + 'whisker_data.csv'):
        
                self.whisker_file.write(",".join(["x{}".format(n) for n in range(24)]) + "," + ",".join(["y{}".format(n) for n in range(24)]) + "\n")
                
        except IOError:
        
            print("Error writing to 'whisker_data.csv'")
            
            raise IOError

        
        ### Load data ###

        try:
        
            if data_type == 'Train':
            
                self.data = pd.read_csv(training_set, names = ['X', 'Y', 'Theta'], header = 0)
                
                self.data_x = self.data['X']
                self.data_y = self.data['Y']
                self.data_body_theta = self.data['Theta']
                self.data_head_theta = self.data['Theta']
                
                #self.data_y = -self.data_y
                
            if data_type == 'Test':

                self.data_file = sp.io.loadmat(test_set)
                
                if data_set == 'Real':
                
                    self.data_x             = self.data_file['data'][0,0][3][0][0][4][:,0] / 10
                    self.data_x             = self.data_x - (np.max(self.data_x) / 2) # put it from (0 -> max) to (-max/2 -> max/2), so whiskeye can start at 0,0
                    self.data_y             = self.data_file['data'][0,0][3][0][0][4][:,1] / 10
                    self.data_y             = self.data_y - (np.max(self.data_y) / 2) # put it from (0 -> max) to (-max/2 -> max/2), so whiskeye can start at 0,0
                    self.data_head_theta    = np.radians(self.data_file['data'][0,0][3][0][0][5][:,0])# - np.pi
                    self.data_body_theta    = np.radians(self.data_file['data'][0,0][3][0][0][6][:,0])
                    self.data_speed         = self.data_file['data'][0,0][3][0][0][7][:,0] / 10
                    self.data_time          = self.data_file['data'][0,0][3][0][0][8][:,0]
                    
                elif data_set == 'VR':
                
                    self.data_x             = self.data_file['data'][0,1][3][0][0][4][:,0] / 10
                    self.data_x             = self.data_x - (np.max(self.data_x) / 2) # put it from (0 -> max) to (-max/2 -> max/2), so whiskeye can start at 0,0
                    self.data_y             = self.data_file['data'][0,1][3][0][0][4][:,1] / 10
                    self.data_y             = self.data_y - (np.max(self.data_y) / 2) # put it from (0 -> max) to (-max/2 -> max/2), so whiskeye can start at 0,0
                    self.data_head_theta    = np.radians(self.data_file['data'][0,1][3][0][0][5][:,0])# - np.pi
                    self.data_body_theta    = np.radians(self.data_file['data'][0,1][3][0][0][6][:,0])
                    self.data_speed         = self.data_file['data'][0,1][3][0][0][7][:,0] / 10
                    self.data_time          = self.data_file['data'][0,1][3][0][0][8][:,0]
                    
                else:
                
                    raise NotImplementedError
                
                self.data = pd.DataFrame({  "X": self.data_x,
                                            "Y": self.data_y,
                                            "Body Theta": self.data_body_theta,
                                            "Head Theta": self.data_head_theta,
                                            "Speed": self.data_speed,
                                            "Time": self.data_time})

            self.data_loaded = True
            
            print("!!! Trajectory loaded !!!")
            print("{} Samples\nRange X: {} to {}\nRange Y: {} to {}\n".format(len(self.data), np.max(self.data_x), np.min(self.data_x), np.max(self.data_y), np.min(self.data_y)))

        except IOError:

            print("Error loading file")
            
            
    ### Subscriber callback functions ###

    def head_packet_callback(self, message):

        self.neck_angles[0:3] = message.neck.data[0:3]

    def body_pose_callback(self, message):

        self.body_pose[0] = message.x

        self.body_pose[1] = message.y

        self.body_pose[2] = message.theta
        
        self.body_pose_file.write("{},{},{},{}\n".format(time.time(), controller.body_pose[0], controller.body_pose[1], controller.body_pose[2]))
        
    def whisker_deflection_callback(self, message):
    
        self.whisker_xy = np.array(message.data)
        
    def camera_callback(self, message):
    
        #image_timestamp = time.time()
        
        #self.timestamp_file.write("{}\n".format(image_timestamp))
        
        #self.image_index = self.image_index + 1
    
        self.image_data = np.frombuffer(message.data, dtype = np.uint8).reshape(message.height, message.width, 3)[:,:,::-1]
        
        #self.body_pose_file.write("{},{},{}\n".format(self.body_pose[0], self.body_pose[1], self.body_pose[2]))
        
        #self.whisker_file.write(",".join(self.whisker_xy.flatten().astype(str)) + "\n") # So you'll get x0...x23 followed by y0...y23 in csv columns
        
                

controller = controller()

kc = kc_interf.kc_whiskeye()

twist_command = Twist()

# Messages

neck_command = Float32MultiArray()

theta_command = Float32MultiArray()

head_pose = Pose2D()

body_pose_message = Float32MultiArray()

# Publishers

pub_neck = rospy.Publisher('/whiskeye/head/neck_cmd', Float32MultiArray, queue_size = 1)

pub_whiskers = rospy.Publisher('/whiskeye/head/theta_cmd', Float32MultiArray, queue_size = 1)

pub_head = rospy.Publisher('/whiskeye/head/pose', Pose2D, queue_size=1)

pub_body = rospy.Publisher('/whiskeye/body/cmd_pos', Float32MultiArray, queue_size = 1)

# Subscribers

rospy.Subscriber('/whiskeye/body/pose', Pose2D, controller.body_pose_callback, queue_size = 1)

rospy.Subscriber('/whiskeye/head/bridge_u', bridge_u, controller.head_packet_callback, queue_size = 1)

rospy.Subscriber('whiskeye/head/cam0/image_raw', Image, controller.camera_callback, queue_size = 1)

rospy.Subscriber('whiskeye/head/xy', Float32MultiArray, controller.whisker_deflection_callback, queue_size = 1)

# Node setup

node = rospy.init_node("Whiskeye_control")

rate = rospy.Rate(controller.update_rate) # 50hz

def loop_through_data():

    while not rospy.is_shutdown():
    
        try:

            for i in range(len(controller.data)):
            
                control()
                
                rate.sleep()
            
        except KeyboardInterrupt:
        
            if not controller.timestamp_file.closed:
            
                controller.timestamp_file.close()
                
            if not controller.body_pose_file.closed:
            
                controller.body_pose_file.close()
                
            if not controller.whisker_file.closed:
            
                controller.whisker_file.close()
                
            raise KeyboardInterrupt
            
        finally:
        
            if not controller.timestamp_file.closed:
            
                controller.timestamp_file.close()
                
            if not controller.body_pose_file.closed:
            
                controller.body_pose_file.close()
                
            if not controller.whisker_file.closed:
            
                controller.whisker_file.close()
                
            sys.exit(0)

def control():

    if controller.index < len(controller.data):

        # If close enough to destination, choose a new destination and go there

        current_x = controller.body_pose[0]

        current_y = controller.body_pose[1]
        
        current_theta = controller.body_pose[2]
        
        next_x = controller.data_x[controller.index]

        next_y = controller.data_y[controller.index]

        #next_body_theta = controller.data_body_theta[controller.index]

        next_head_theta = 0#controller.data_head_theta[controller.index]
        
        next_body_theta = controller.data_head_theta[controller.index]
        
        controller.destination = [next_x, next_y, next_body_theta]

        body_pose_message.data = controller.destination

        pub_body.publish(body_pose_message)
        
        if controller.index > 0:

            print('\33[1K\33[1F\33[1K\33[1F\33[1K\33[1F\33[1K\33[1F') # Delete prior 4 lines of console (including ending \n to prevent flickering)
        
        print("Sample: {}".format(controller.index + 1))
        print("Current body pose: {}\nTarget body pose:  {}".format(np.round(controller.body_pose, 4), np.round(controller.destination, 4)), end = '\n')

        ## find head pose in body frame

        # lay into model

        kc.setConfig(controller.neck_angles)

        # get location of head centre and fovea in HEAD

        cen_HEAD = np.array([0.0, 0.0, 0.0])

        fov_HEAD = np.array([0.1, 0.0, 0.0])

        # transform to BODY

        cen_BODY = kc.changeFrameAbs(

        kc_interf.KC_FRAME_HEAD,

        kc_interf.KC_FRAME_BODY,

        cen_HEAD)

        fov_BODY = kc.changeFrameAbs(

        kc_interf.KC_FRAME_HEAD,

        kc_interf.KC_FRAME_BODY,

        fov_HEAD)

        # recover head yaw angle

        d = fov_BODY - cen_BODY

        yaw = np.array([np.arctan2(d[1], d[0])])

        # and location of head center in BODY (x, y)

        xy = cen_BODY[0:2]

        # cast head into world frame and publish

        #head_pose.theta = controller.body_pose[2] + yaw

        head_pose.x = controller.body_pose[0]+ (xy[0] * math.cos(controller.body_pose[2]))

        head_pose.y = controller.body_pose[1]+ (xy[0] * math.sin(controller.body_pose[2]))

        neck_command.data = [0, 0, next_head_theta]

        pub_neck.publish(neck_command)

        # shake those whiskers
        #x = np.sin(controller.data_time[controller.index]*controller.fs*2*np.pi*controller.whisk_rate)
        
        controller.prior_whisker_theta = controller.current_whisker_theta
        
        controller.current_whisker_theta = np.sin(controller.index * controller.fs * 2*np.pi * controller.whisk_rate)
        
        theta_command.data = np.ones(shape = (24,)) * controller.current_whisker_theta
        
        pub_whiskers.publish(theta_command)
        
        # If peak whisker protraction is reached, save data to file
        
        if controller.prior_whisker_theta > controller.current_whisker_theta and controller.peaked == False:
        
            controller.peaked = True 
        
            #print("Gradient flip: {} > {}".format(np.round(controller.prior_whisker_theta, 4), np.round(controller.current_whisker_theta , 4)))
            
            controller.whisker_file.write(",".join(controller.whisker_xy.flatten().astype(str)) + "\n") # So you'll get x0...x23 followed by y0...y23 in csv columns
            
            if controller.image_data is not None:
            
                controller.image_index = controller.image_index + 1
            
                #controller.image_timestamp = time.time()
            
                controller.timestamp_file.write("{}\n".format(time.time()))
            
                cv2.imwrite(controller.image_folder + "{}.png".format(controller.image_index), controller.image_data)
            
        elif controller.prior_whisker_theta < controller.current_whisker_theta:
        
            controller.peaked = False
        

        controller.index = controller.index + 1
        
    else:
    
        print("\n\nData file exhausted")
        
if __name__ == "__main__":

    loop_through_data()
