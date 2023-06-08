import numpy as np
from custom_env import CustomEnv
import matplotlib.pyplot as plt
import torch
from collections import deque
import random 
import gym
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_LENGTH = 10000
EPSILON = 1.0
MIN_EPSILON = 0.1 
EPSILON_DECAY = 0.001 
EPSILON_DECAY_ITER = 100 
LEARNING_RATE = 0.0005
UPDATE_FREQ = 4 
TARGET_NETWORK_UPDATE_FREQ = 1000 
N_ACTIONS = 8

env = CustomEnv(gui=1,n_actions=N_ACTIONS)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
env.seed(0)

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 4
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,N_ACTIONS)   
        )
    def forward(self,x):
        return self.net(x)
 
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype= torch.float32)

        q_values = self(obs_t.unsqueeze(0))

        max_q_index= torch.argmax(q_values,dim=1)[0]

        action = max_q_index.detach().item()

        return action
    
rew_buffer =  deque(maxlen = 200)
replay_buffer = deque(maxlen= BUFFER_LENGTH)
plot_buffer = deque()
device = 'cpu'


Online_Net = Network()
Target_Net = Network()
# Online_Net = torch.load("v_4500_decay100_online_net.pth")
# Online_Net.eval()
# Target_Net = torch.load("v_4500_decay100_target_net.pth")
# Target_Net.eval()


Online_Net.to(device)
Target_Net.to(device)
Target_Net.load_state_dict(Online_Net.state_dict())
optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=Online_Net.parameters())

for episode in range(100):
    env.reset()
    done = False
    while not done:  
        old_state= env.high_level_state()
        action = np.random.randint(N_ACTIONS)

        _, reward, done, _  = env.step(action)
        new_state = env.high_level_state()

        
        transition = (old_state, action,reward, done, new_state) 
        replay_buffer.append(transition)   

stepNo=0
for episode in range(4001):  
    episode_reward = 0.0  
    env.reset()
    x =env.state()

    done = False
    while not done: 
        
        stepNo=stepNo+1
        old_state= env.high_level_state()
        random_sample = random.random()
        action =0 
        if random_sample <= EPSILON :
            action  =np.random.randint(N_ACTIONS)
        else:
            action = Online_Net.act(torch.as_tensor(old_state).to(device) ) 
        _, reward, done, _  = env.step(action)
        new_state= env.high_level_state()
        transition = (old_state, action,reward, done, new_state)
        replay_buffer.append(transition)
        episode_reward  = episode_reward + reward
        if done:
            rew_buffer.append(episode_reward)
            plot_buffer.append(np.mean(rew_buffer) )

        if stepNo % UPDATE_FREQ == 0 or done:
            transitions = random.sample(replay_buffer,BATCH_SIZE)

            sample_old_state =np.array(list(t[0] for t in transitions))
            sample_action   = np.array(list(t[1] for t in transitions))
            sample_reward = np.array(list(t[2] for t in transitions))
            sample_done = np.array(list(t[3] for t in transitions))
            sample_new_state = np.array(list(t[4] for t in transitions))

            sample_old_state_t  = torch.as_tensor(sample_old_state, dtype = torch.float32).to(device) 
            sample_action_t = torch.as_tensor(sample_action, dtype = torch.int64).unsqueeze(-1).to(device) 
            sample_reward_t = torch.as_tensor(sample_reward, dtype = torch.float32).unsqueeze(-1).to(device) 
            sample_done_t = torch.as_tensor(sample_done, dtype = torch.float32).unsqueeze(-1).to(device) 
            sample_new_state_t = torch.as_tensor(sample_new_state, dtype = torch.float32).to(device) 

            target_q_values = Target_Net(sample_new_state_t)

            max_target_q_values = target_q_values.max(dim=1,keepdim=True)[0] 

            targets = sample_reward_t +GAMMA* (1- sample_done_t) * max_target_q_values   
        
            q_values = Online_Net(sample_old_state_t)

            action_q_values = torch.gather(input = q_values , dim =1, index = sample_action_t )

            loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if stepNo % TARGET_NETWORK_UPDATE_FREQ == 0 :
            Target_Net.load_state_dict(Online_Net.state_dict())

        if stepNo % EPSILON_DECAY_ITER == 0 :
            if EPSILON > MIN_EPSILON:
                EPSILON = EPSILON - EPSILON_DECAY


    if episode % 500 == 0  :
        with open("e_" +str(episode) +"plot_buffer_4000_222_dqn.npy", 'wb') as f:
            np.save(f, plot_buffer)
        with open("e_" +str(episode) +"replay_buffer_4000_222_dqn.npy", 'wb') as f:
            np.save(f, replay_buffer)
        with open("e_" +str(episode) +"rew_buffer_4000_222_dqn.npy", 'wb') as f:
            np.save(f, rew_buffer)
        with open("e_" +str(episode) +"epsilon_4000_222_dqn.npy", 'wb') as f:
            arr  = np.zeros(1)
            arr[0] = EPSILON
            np.save(f, arr)
            
        torch.save(Online_Net , "e_" +str(episode) +"_decay100_online_net_222.pth")
        torch.save(Target_Net , "e_" +str(episode)+"_decay100_target_net_222.pth")
    if episode % 100 == 0 :
        print()
        print("episode",episode )
        print("avg rew : "   ,np.mean(rew_buffer) )
        print("len rew : "   ,len(rew_buffer) )
        print(EPSILON)
        print()


plt.plot(plot_buffer)
plt.title("Running Avarage Of Rewards")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig('dqn_pybullet_decay100_222.png')
plt.show()














