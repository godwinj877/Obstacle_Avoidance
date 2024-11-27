# !/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from math import *
from collections import deque
import os

REPLAY_BUFFER_SIZE = 100000

MODEL_PATH = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DQN/models/"

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
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

class DQN(nn.Module):
    def __init__(self, n_observations=12, n_actions=8):
        super().__init__()
        self.layer_1 = nn.Linear(n_observations, 512)
        nn.init.xavier_uniform_(self.layer_1.weight)
        self.layer_1.bias.data.fill_(0.01)
        self.layer_2 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.layer_2.weight)
        self.layer_2.bias.data.fill_(0.01)
        self.layer_3 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.layer_3.weight)
        self.layer_3.bias.data.fill_(0.01)
        self.layer_4 = nn.Linear(512, n_actions)
        nn.init.xavier_uniform_(self.layer_4.weight)
        self.layer_4.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        x = F.leaky_relu(self.layer_3(x))
        action = F.tanh(self.layer_4(x))
        return action

class Agent:
    def __init__(self, state_dim, action_dim, alpha= 0.001, beta=0.95,
                 gamma = 0.95, max_size = 1e8, tau = 0.005,
                 batch_size = 16):
        self.mem_size = 0 
        self.gamma = gamma
        self.tau = tau
        self.dqn = DQN(state_dim, action_dim).to(device)
        self.batch_size = batch_size
        self.target_dqn = DQN(state_dim, action_dim).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr = alpha)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    def choose_action(self, current_state, current_action, current_reward):
        print("Choosing action")
        current_state = torch.tensor(current_state, dtype=torch.float32)
        current_action = torch.tensor(current_action, dtype=torch.float32)
        current_reward = torch.tensor(current_reward, dtype=torch.float32)

        state = torch.cat([current_state, torch.argmax(current_action).unsqueeze(0), current_reward.unsqueeze(0)], axis=0)
        print(state)
        with torch.no_grad():
            q_values = self.dqn(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.mem_size += 1
    
    def train(self, state, action, reward, next_state, done):
        if self.mem_size < self.batch_size:
            return
        
        print("Started Learning")

        current_state, action, reward, next_state, done = self.replay_buffer.sample_batch(self.batch_size)

        current_state = torch.tensor(current_state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        predicted_q_values = self.dqn(current_state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            target_q_values_next = self.target_dqn(next_state).max(1)[0].unsqueeze(1)
            target_q_values = reward.unsqueeze(1) + self.gamma * target_q_values_next.max(1)[0].unsqueeze(1)

        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def save_models(self, time, model_path=MODEL_PATH):
        print("Saving models...")
        torch.save(self.dqn.state_dict(), model_path+f'{time}/dqn/dqn_{self.replay_buffer.num_experiences}.pt')
        torch.save(self.target_dqn.state_dict(), model_path+f'{time}/target_dqn/target_dqn_{self.replay_buffer.num_experiences}.pt')

    def load_models(self, model_path=MODEL_PATH):
        print("Loading models...")
        self.dqn.load_state_dict(torch.load(self.get_latest_model(model_path+'/dqn')))
        self.target_dqn.load_state_dict(torch.load(self.get_latest_model(model_path+'/target_dqn')))
    
    def get_latest_model(self, PATH):
        model_files = os.listdir(PATH)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(PATH, x)))
        last_saved_model_path = os.path.join(PATH, model_files[-1])
        print("Last Saved Model: ", last_saved_model_path)
        return last_saved_model_path