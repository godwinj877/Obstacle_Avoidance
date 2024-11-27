# !/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from collections import deque
import os

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DDPG/models/"

ACTOR_PATH = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DDPG/models/actor"
CRITIC_PATH = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DDPG/models/critic"
TARGET_ACTOR_PATH = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DDPG/models/target_actor"
TARGET_CRITIC_PATH = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DDPG/models/target_critic"

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.num_experiences = 0

    def add(self, state, action, reward, next_state, done):
        print("Experience: ", self.num_experiences+1)
        if self.num_experiences < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
    def count(self):
        return self.num_experiences
    
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, name):
        super(ActorNetwork, self).__init__()
        self.model_name = name

        self.layer_1 = nn.Linear(state_dim, 512)
        nn.init.xavier_uniform_(self.layer_1.weight)
        self.layer_1.bias.data.fill_(0.01)
        self.layer_2 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.layer_2.weight)
        self.layer_2.bias.data.fill_(0.01)
        self.layer_3 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.layer_3.weight)
        self.layer_3.bias.data.fill_(0.01)
        self.layer_4 = nn.Linear(512, action_dim)
        nn.init.xavier_uniform_(self.layer_4.weight)
        self.layer_4.bias.data.fill_(0.01)
        # self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        x = F.leaky_relu(self.layer_3(x))
        action = F.tanh(self.layer_4(x))
        # mean = torch.tanh(self.layer_4(x))
        # std = F.softplus(self.std)
        # return mean, std
        return action
    
    def sample_normal(self, state):
        mean, std = self.forward(state)
        normal_dist = torch.distributions.normal.Normal(mean, std)
        action = normal_dist.sample()
        action = torch.tanh(mean + std * action)
        return action
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, name):
        super(CriticNetwork, self).__init__()
        self.model_name = name

        self.layer_1 = nn.Linear(state_dim + action_dim, 512)
        nn.init.xavier_uniform_(self.layer_1.weight)
        self.layer_1.bias.data.fill_(0.01)
        self.layer_2 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.layer_2.weight)
        self.layer_2.bias.data.fill_(0.01)
        self.layer_3 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.layer_3.weight)
        self.layer_3.bias.data.fill_(0.01)
        self.layer_4 = nn.Linear(512, 1)
        nn.init.xavier_uniform_(self.layer_4.weight)
        self.layer_4.bias.data.fill_(0.01)

    def forward(self, state, action):
        # print("State:", state)
        # print("Action:", action)
        x = torch.cat((state, action), dim=1)
        x = x.squeeze()
        x = x.to(device)
        # print("x:", x)
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        x = F.leaky_relu(self.layer_3(x))
        q_value = self.layer_4(x)

        return q_value

# class ActorNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, name):
#         super(ActorNetwork, self).__init__()
#         self.model_name = name
#         self.lstm = nn.LSTM(state_dim, 400, batch_first=True)
#         self.fc1 = nn.Linear(400, 300)
#         self.fc2 = nn.Linear(300, action_dim)
#         self.relu = nn.ReLU()
#         self.batch_norm1 = nn.BatchNorm1d(400)
#         self.batch_norm2 = nn.BatchNorm1d(300)
        
#     def forward(self, state):
#         lstm_out, _ = self.lstm(state)
#         x = self.batch_norm1(lstm_out[:, -1, :])  # Taking the last output of LSTM
#         x = self.relu(x)
#         x = self.fc1(x)
#         x = self.batch_norm2(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return torch.tanh(x)  # Applying tanh activation for action output

# class CriticNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, name):
#         super(CriticNetwork, self).__init__()
#         self.model_name = name
#         self.lstm = nn.LSTM(state_dim, 400, batch_first=True)
#         self.fc1 = nn.Linear(400 + action_dim, 300)
#         self.fc2 = nn.Linear(300, 1)
#         self.relu = nn.ReLU()
#         self.batch_norm1 = nn.BatchNorm1d(400)
#         self.batch_norm2 = nn.BatchNorm1d(300)
        
#     def forward(self, state, action):
#         lstm_out, _ = self.lstm(state)
#         x = self.batch_norm1(lstm_out[:, -1, :])  # Taking the last output of LSTM
#         x = self.relu(x)
#         x = torch.cat([x, action], dim=1)  # Concatenating action with LSTM output
#         x = self.fc1(x)
#         x = self.batch_norm2(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x 
     
class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim, batch_size=32, gamma=0.9):
        super(DDPGAgent, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.mem_size = 0

        self.actor = self.build_actor(state_dim, action_dim, 'actor').to(device)
        self.critic_1 = self.build_critic(state_dim, action_dim, 'critic_1').to(device)
        self.critic_2 = self.build_critic(state_dim, action_dim, 'critic_2').to(device)


        self.target_actor = self.build_actor(state_dim, action_dim, 'target_actor').to(device)
        self.target_critic_1 = self.build_critic(state_dim, action_dim, 'target_critic_1').to(device)
        self.target_critic_2 = self.build_critic(state_dim, action_dim, 'target_critic_2').to(device)

        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=0.0001)
        self.critic_1_optimizer = optim.Adam(params=self.critic_1.parameters(), lr=0.0001)
        self.critic_2_optimizer = optim.Adam(params=self.critic_2.parameters(), lr=0.0001)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.soft_update(1)

    def build_actor(self, state_dim, action_dim, name):
        return ActorNetwork(state_dim=state_dim, action_dim=action_dim, name=name)
    
    def build_critic(self, state_dim, action_dim, name):
        return CriticNetwork(state_dim=state_dim, action_dim=action_dim, name=name)
    
    def train(self, state, action, reward, next_state, done):
        if self.mem_size < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample_batch(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        # Update critic
        print("Updating critic network")
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            # print("Next Action: ", next_action)
            print("Next Action: ", torch.argmax(next_action))
            critic_1 = self.target_critic_1(next_state, next_action)
            critic_2 = self.target_critic_2(next_state, next_action)
            q_next = torch.minimum(critic_1, critic_2)
            target_q = reward + self.gamma*(1-done)*q_next
        
        q_current_1 = self.critic_1(state, action)
        q_current_2 = self.critic_2(state, action)
        critic_loss_1 = F.mse_loss(q_current_1, target_q)
        critic_loss_2 = F.mse_loss(q_current_2, target_q)
        print("Critic loss: ", min(critic_loss_1, critic_loss_2))

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Update actor
        print("Updating actor network")
        actor_loss = torch.mean(-torch.min(self.critic_1(state, self.actor(state)), self.critic_2(state, self.actor(state))))
        print("Actor loss: ", actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()   

        self.soft_update(0.01)                               

    def soft_update(self, tau):
        print("Updating target network")
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()
        return action.flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.mem_size += 1

    def save_models(self, time, model_path=MODEL_PATH):
        print("Saving models...")
        torch.save(self.actor.state_dict(), model_path+f'{time}/actor/actor_{self.replay_buffer.num_experiences}.pt')
        torch.save(self.target_actor.state_dict(), model_path+f'{time}/target_actor/target_actor_{self.replay_buffer.num_experiences}.pt')
        torch.save(self.critic_1.state_dict(), model_path+f'{time}/critic_1/critic_{self.replay_buffer.num_experiences}.pt')
        torch.save(self.critic_2.state_dict(), model_path+f'{time}/critic_2/critic_{self.replay_buffer.num_experiences}.pt')
        torch.save(self.target_critic_1.state_dict(), model_path+f'{time}/target_critic_1/target_critic_{self.replay_buffer.num_experiences}.pt')
        torch.save(self.target_critic_2.state_dict(), model_path+f'{time}/target_critic_2/target_critic_{self.replay_buffer.num_experiences}.pt')

    def load_models(self, model_path=MODEL_PATH):
        print("Loading models...")
        self.actor.load_state_dict(torch.load(self.get_latest_model(model_path+'/actor'), map_location=device))
        self.target_actor.load_state_dict(torch.load(self.get_latest_model(model_path+'/target_actor'), map_location=device))
        self.critic_1.load_state_dict(torch.load(self.get_latest_model(model_path+'/critic_1'), map_location=device))
        self.critic_2.load_state_dict(torch.load(self.get_latest_model(model_path+'/critic_2'), map_location=device))
        self.target_critic_1.load_state_dict(torch.load(self.get_latest_model(model_path+'/target_critic_1'), map_location=device))
        self.target_critic_2.load_state_dict(torch.load(self.get_latest_model(model_path+'/target_critic_2'), map_location=device))

    def get_latest_model(self, PATH):
        model_files = os.listdir(PATH)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(PATH, x)))
        last_saved_model_path = os.path.join(PATH, model_files[-1]) #4
        print("Last Saved Model: ", last_saved_model_path)
        return last_saved_model_path