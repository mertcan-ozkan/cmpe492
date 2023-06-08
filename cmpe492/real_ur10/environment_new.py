import time
import torch
import math
import torchvision.transforms as transforms
import numpy as np
import time
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ur_ikfast import ur_kinematics
from scipy.spatial.transform import Rotation as R
import tkinter as tk
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
from gym import spaces
import timeit
from collections import deque
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

rospy.init_node('gui', anonymous=True)
pub = rospy.Publisher('radian', Float64MultiArray, queue_size=10)

joint_order = [2, 1, 0, 3, 4, 5]
ur10_arm = ur_kinematics.URKinematics('ur10')

tool_length = 0.2 #distance from ur10 end and tool center

TASKSPACE_LIMITS = {'x':[-1.29, -0.62],'y':[-0.35, 0.6],'z':[0.24, 0.7]} 
MAX_LIN_SPEED_TOOL = 0.05 # m/s
ROS_INTERFACE_RATE = 10 # Hz

dt = 1/ROS_INTERFACE_RATE
rate = rospy.Rate(ROS_INTERFACE_RATE) # 10hz   
def collect(current_state, current_pose):
    output_file = open("traj.txt", 'a')
    output_file.write(str(current_state)+'\n')
    output_file.flush()
    rate.sleep()
    output_file.close()
def return_to_pos():
    input_file = open("traj.txt", 'r')
    # get initial state from the file:
    traj = []
    for state in input_file:
        state= [float(i) for i in state[1:-2].split(',')]
        state = [state[i] for i in joint_order]
        traj.append(state)
    # execute trajectory in reverse order
    for state in reversed(traj):
        pub.publish(Float64MultiArray(data=state+[dt]))
        rate.sleep()
    input_file.close()
    open('traj.txt', 'w').close()  ## clear file 
    
def init_position():
    open('traj.txt', 'w').close()  ## clear file 
    input_file = open('init.txt', 'r')
    # get initial state from the file:
    current_state, _ = get_current_state(return_pose=True,collect_traj=False)
    print(current_state)
    starting_position = [-0.0008853117572229507, -1.5727275053607386, 0.0065135955810546875, -1.5761678854571741, -0.0015085379229944351, 0.0006707421271130443]
    for i in range(len(current_state)):
        if i ==0 :continue
        if abs (current_state[i] -starting_position[i] ) > 0.1:
            while True:
                current_state, _ = get_current_state(return_pose=True,collect_traj=False)
                print(current_state)
                print("not in inital position!!!!!!!")
                time.sleep(3)
    print("moving to starting position...")
    time.sleep(4)
    traj = []
    for state in input_file:
        state= [float(i) for i in state[1:-2].split(',')]
        state = [state[i] for i in joint_order]
        traj.append(state)
    for state in traj:
        pub.publish(Float64MultiArray(data=state+[dt]))
        rate.sleep()

    input_file.close()
def get_current_state(return_pose=True,collect_traj=True):
    current_state = rospy.wait_for_message("/joint_states", JointState)
    current_state_lst = [current_state.position[i] for i in joint_order]
    pose_matrix=None
    if return_pose:
        pose_matrix= ur10_arm.forward(current_state_lst, 'matrix')
    if collect_traj :
        collect(current_state_lst,pose_matrix )
    return current_state_lst, pose_matrix
##dx=0.01
def MoveArm(target_pose, current_pose=None, current_state=None, dx=0.01, time_from_start=1):
    # theta target is ['x','y','Z'] first two are extrinsic euler angles, last one is intrinsic
    # target_pose and current_pose are 3x4 matrices !!!
    print("moving ... ")
    dtheta=(1/2*np.pi)/time_from_start

    if current_pose is None or current_state is None:
        current_state, current_pose = get_current_state(return_pose=True,collect_traj =True)

    p_initial = current_pose[:3,-1]
    p_target = target_pose[:3,-1]
    target_distance = np.linalg.norm(p_target-p_initial)
    target_speed = target_distance/time_from_start  
    
    # # safety check - if speed demand is within limits
    # if target_speed>MAX_LIN_SPEED_TOOL:
    #     print("speed exceeds limit! Target speed:   " ,target_speed )
    #     return
    
    n_steps=max(2, int(abs(target_distance)/dx)+1)

    # dont allow to go beyond task space limits!

    for idx, axis in enumerate(['x','y','z']):
        p_target[idx] = np.clip(p_target[idx], TASKSPACE_LIMITS[axis][0], TASKSPACE_LIMITS[axis][1])


    p_vals = np.linspace(p_initial, p_target, n_steps)[1:]

    # get euler angles from pose matrix
    current_euler = R.from_matrix(current_pose[:3,:3]).as_euler('xyz', degrees=False)
    target_euler = R.from_matrix(target_pose[:3,:3]).as_euler('xyz', degrees=False)
    idx_max_angle_change = np.argmax(np.abs(target_euler-current_euler))
    n_steps2 = max(2, int(abs(target_euler[idx_max_angle_change]-current_euler[idx_max_angle_change])/dtheta)+1)
    n_steps = max(n_steps, n_steps2)
    
    int_eulers = np.linspace(current_euler, target_euler, n_steps)[1:] # intermediate euler angles
    p_vals = np.linspace(p_initial, p_target, n_steps)[1:]

    # rospy.loginfo("intermediate_p_vals:"+np.array2string(p_vals))
    data_to_send = Float64MultiArray()

    new_pose = current_pose
    for p_target_i, euler_i in zip(p_vals, int_eulers):
        new_pose[:3,-1] = p_target_i
        new_pose[:3,:3] = R.from_euler('xyz', euler_i, degrees=False).as_matrix()
        new_joint_pos = ur10_arm.inverse(new_pose, False, q_guess=current_state)
        if new_joint_pos is not None:
            # rospy.loginfo("going to "+np.array2string(new_pose))
            data_to_send.data = [new_joint_pos[i] for i in joint_order]+[time_from_start/n_steps]
            pub.publish(data_to_send)
        else:
            rospy.loginfo("cant find solution for the orientation:"+np.array2string(euler_i))
    
def compute_target_pose(current_pose, action, w_x =0.0 , w_y= 0.0 ,w_Z=0.0 , dt = 1.0):
    action = np.append(action, 0.0)
    print(action)
    #compute the target position
    p_initial = current_pose[:3,-1]
    p_target = p_initial + action
    # compute the target orientation
    R_initial = current_pose[:3,:3]
    R_target = R_initial
    R_target = np.dot(R_target, R.from_euler('x', w_x*dt, degrees=True).as_matrix())
    R_target = np.dot(R_target, R.from_euler('y', w_y*dt, degrees=True).as_matrix())
    R_target = np.dot(R.from_euler('Z', w_Z*dt, degrees=True).as_matrix(), R_target)
    target_pose = np.zeros((3,4), dtype=np.float32)
    target_pose[:3,:3] = R_target
    target_pose[:,-1] = p_target
    return target_pose

v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
w_Z = 0.0
w_x = 0.0
w_y = 0.0
dT  = 1.0
class Env(gym.Env):
    def __init__(self):
        self._t = 0
        self.episode_reward = 0
        self._goal_thresh = 0.03
        self._max_timesteps = 150
        self.n_actions = 8
        self._actions = [ [0.05,0.0 ],
                        [-0.05,0.0],
                        [0.0,0.05] ,
                        [0.0,-0.05],
                        [0.05/math.sqrt(2) , 0.05/math.sqrt(2) ],
                        [-0.05/math.sqrt(2) ,0.05/math.sqrt(2) ],
                        [0.05/math.sqrt(2) ,-0.05/math.sqrt(2) ],
                        [-0.05/math.sqrt(2),-0.05/math.sqrt(2) ]
                        ]
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low= np.array([-5,-5,-5,-5]),high= np.array([+5,+5,+5,+5]),shape=(4,) ,dtype=np.float64 )
        self.goals=np.zeros((4,3))
        self.goal_no =0 
        self.time_buffer =  deque()
        self.episode_steps = deque()
        self.rew_buffer =  deque()
        self.iteration_no = 0
        self.episode_limit=15
        self.init_goal_pos()
        init_position() 
        self.start_time =  timeit.default_timer()
    def reset(self):
        return_to_pos() 
        self._t = 0
        self.start_time =  timeit.default_timer()
        return self.high_level_state()

    def init_goal_pos(self):

        self.goals[0] =   [-1,
         -0.1,
        1.025]
        self.goals[1] =  [-0.7,
         -0.1,
        1.025]
        self.goals[2] =  [-0.7,
        0.4,
        1.025]
        self.goals[3] =  [-1,
        0.4,
        1.025]
        # self.goals[0] =   [-1,
        #  0.15,
        # 1.025]
        # self.goals[1] =  [-0.7,
        #  -0.1,
        # 1.025]
        # self.goals[2] =  [-0.7,
        # 0.4,
        # 1.025]
    
        self.goal_pos = self.goals[self.goal_no] 
        return self.goal_pos

    def step(self, action_id):
        action = self._actions[int(action_id)]
        print("action:  "   ,action ) 
        if self._t == 0 :
            for i in range(8):
                print(self._actions[int(i)])
        current_state, current_pose  =   get_current_state(return_pose=True)
        target_pose = compute_target_pose(current_pose,action, w_x, w_y, w_Z, dt=dT)
        MoveArm(target_pose, current_state=current_state, current_pose=current_pose, time_from_start=dT)
        rate.sleep()
        
        self._t = self._t + 1  
        state = self.high_level_state()
        reward = self.reward()
        self.episode_reward = self.episode_reward+reward
        
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        done = terminal or truncated
        if done:
            self.rew_buffer.append(self.episode_reward)
            self.episode_steps.append(self._t)
            self.episode_reward = 0
            duration = timeit.default_timer() - self.start_time 
            self.time_buffer.append(duration)
            self.iteration_no = self.iteration_no +1
            if self.iteration_no == self.episode_limit :

                with open('ppo_rewards_env1_mj.npy', 'wb') as ff:
                    np.save(ff, self.rew_buffer)
                with open('ppo_time_env1_mj.npy', 'wb') as ff:
                    np.save(ff, self.time_buffer)
                with open('ppo_episode_steps_env1_mj.npy', 'wb') as ff:
                    np.save(ff, self.episode_steps)

                print((self.rew_buffer))
                print((self.time_buffer)) 
                print((self.episode_steps))
                plt.plot(self.episode_steps)
                plt.savefig('ppo_episode_steps_env1_mj.png')
                plt.close()                
                plt.plot(self.rew_buffer)
                plt.savefig('ppo_rewards_env1_mj.png')
                plt.close()
                plt.plot(self.time_buffer)
                plt.savefig('ppo_time_env1_mj.png')
                plt.close()
                print("done")
                time.sleep(5)
            return state, reward, done , {}
        else:

            return state, reward, done , {}
         

    def high_level_state(self):
        _, current_pose  =   get_current_state(return_pose=True)
        p_initial = current_pose[:3,-1]
        r=np.zeros(2)
        r[0] = p_initial[0]
        r[1] = p_initial[1]
        self.ee_pos =  r
        return np.concatenate([self.ee_pos[:2], self.goal_pos[:2]])

    def reward(self):
        distance = -1*np.sqrt(np.sum((self.ee_pos[:2] - self.goal_pos[:2])**2))
        # print ("ee pos :   ", self.ee_pos[:2])
        # print("goal pos :  ", self.goal_pos[:2])
        # print("reward:   ",distance )
        distance = -1*np.sqrt(np.sum((self.ee_pos[:2] - self.goal_pos[:2])**2))
        return distance 

    def is_terminal(self):
        ee_pos = self.ee_pos[:2]
        goal_pos = self.goals[self.goal_no][:2] 
        print("ee_pos : ," , ee_pos)
        print("goal_pos : ," , goal_pos)
        print (np.sqrt(np.sum((ee_pos - goal_pos)**2)) )
        if self.goal_no == len(self.goals) -1:
            if np.sqrt(np.sum((ee_pos - goal_pos)**2)) < self._goal_thresh: 
                 self.goal_no =0 
                 self.goal_pos = self.goals[self.goal_no]
                 print("self.iteration_no    ",self.iteration_no)
            return np.sqrt(np.sum((ee_pos - goal_pos)**2)) < self._goal_thresh
        else:
            if  np.sqrt(np.sum((ee_pos - goal_pos)**2)) < self._goal_thresh:
                print("                     goal increase")
                self.goal_no = self.goal_no +1
                self.goal_pos = self.goals[self.goal_no]
                return False
            else:
                return False


    

    def is_truncated(self):
        if self._t >= self._max_timesteps :
            print("TRUNCATED")
        return self._t >= self._max_timesteps
    

