import copy
from collections import deque
import numpy as np
import pybullet
import pybullet_data
import time
import utils
import manipulators
import torch
import torchvision.transforms as transforms
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import math
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class CustomEnv(gym.Env):
    def __init__(self, gui=0,n_actions=8 ):
        self._p = utils.connect(gui)
        
        self.gui = gui
        self._reset()
        x = -1.4
        y = 0.15    
        self.ee_pos =   np.array([0.79798992+x, 0.17399998+y, 0.47]) 
        self._t = 0
        self.rew_buffer =  deque([0,0],maxlen = 200)  ### for plot 
        self.len_buffer =  deque([0,0])
        self.plot_buffer =  deque([0,0])
        self.episode_reward =0 
        self.time = 0


        self._goal_thresh = 0.02
        self._max_timesteps = 50
        self._n_actions = n_actions
        self._actions =[ [0.05,0.0 ],
                        [-0.05,0.0],
                        [0.0,0.05] ,
                        [0.0,-0.05],
                        [0.05/math.sqrt(2) , 0.05/math.sqrt(2) ],
                        [-0.05/math.sqrt(2) ,0.05/math.sqrt(2) ],
                        [0.05/math.sqrt(2) ,-0.05/math.sqrt(2) ],
                        [-0.05/math.sqrt(2),-0.05/math.sqrt(2) ]
                        ]
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low= np.array([-5,-5,-5,-5]),high= np.array([+5,+5,+5,+5]),shape=(4,) ,dtype=np.float64 )
    def _reset(self):
        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")
        x = -1.4
        y = 0.15
        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0.+x, 0.+y, 0.4], ik_idx=10)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 8, self.agent.id, 9,
                                                    jointType=self._p.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
        self._p.changeConstraint(mimic_constraint, gearRatio=-1, erp=0.1, maxForce=50)

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.950, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def state_obj_poses(self):
        N_obj = len(self.obj_dict)
        pose = np.zeros((N_obj, 7), dtype=np.float32)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            pose[i][3:] = quaternion
        return pose

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()

    def __del__(self):
        self._p.disconnect()

    def reset(self):
        self._reset()
        x = -1.4
        y = 0.15
        self.ee_pos =  np.array([0.79798992+x, 0.17399998+y, 0.47]) 
        self.obj_dict = {}
        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)
        self._t = 0
        self.agent.close_gripper(1, sleep=True)


        return np.concatenate([self.ee_pos[:2], self.goal_pos[:2]])

    def reset_objects(self):
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}
        self.init_objects()
        self._step(240)

    def init_objects(self):
        obj_type = self._p.GEOM_CYLINDER
        x =np.random.uniform(-1.1 , -0.60)
        y = np.random.uniform(-0.2 ,0.50)
        self.goal_pos = [x, y, 0.6]
        rotation = [0, 0, 0]
        
        self.obj_dict[0] = utils.create_object(p=self._p, obj_type=obj_type, size=[0.05, 0.005], position=self.goal_pos,
                                               rotation=rotation, color=[0.2, 1.0, 0.2, 1], mass=0.5)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.5, 0.0, 1.5], target_position=[0.9, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, action_id, sleep=False):
        action = self._actions[action_id]
        self.ee_pos[:2]  = np.clip(self.ee_pos[:2]  + action, [-1.1, -0.2 ], [-0.60, 0.50])
       
        self.agent.set_cartesian_position(self.ee_pos,
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
        self._t = self._t + 1        
        h_state = self.high_level_state()
        reward = self.reward()
        self.episode_reward = self.episode_reward+reward
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        if self.time == 4000:  
                   
                with open('plot_buffer_4000_111_ppo.npy', 'wb') as f:
                    np.save(f, self.plot_buffer)
                with open('len_buffer_4000_111_ppo.npy', 'wb') as ff:
                    np.save(ff, self.len_buffer)

                plt.plot(self.plot_buffer)
                plt.title("Running Avarage Of Rewards")
                plt.ylabel("Rewards")
                plt.savefig('pybullet_ppo_4000_111_ppo.png')
                

               
        if  terminal or truncated  :   
          
            self.time  =self.time +1
            self.rew_buffer.append(self.episode_reward)
            self.len_buffer.append(self._t)
            self.episode_reward =0 
            self.plot_buffer.append(np.mean(self.rew_buffer))

            return h_state, reward, True, {}
        else: 
            return h_state, reward, False, {}
        

    def high_level_state(self):
        return np.concatenate([self.ee_pos[:2], self.goal_pos[:2]])

    def reward(self):

        distance =   -1*np.sqrt(np.sum((self.ee_pos[:2] - self.goal_pos[:2])**2))
        return distance

    def is_terminal(self):
        
        return np.sqrt(np.sum((self.ee_pos[:2] - self.goal_pos[:2])**2)) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps