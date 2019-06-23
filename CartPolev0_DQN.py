'''Austin Nguyen ~ 6/13/2019 ~ Problem #1
CartPolev0 ~ DQ Network
Results: Very solid duration for each iteration. Translational motion was harder to temper than angular motion.
Changes to consider: playing with the LR for more precise convergence (almost slight oscillations right now) and play with hid depth width
Inspiration and help from @MorvanZhou'''

'''
Intuition:
    - Off policy technique that maps state and action pair to a Q value
    - Based on parameter, either greedily or probabilistically chooses next action based on that Q value
    - Training/loss uses the Bellman update equation
Takeaways:
    - Random individual sampling from experience replay gives much better results than just than sampling in contiguous chunks
    - np.random.choice is very useful
    - Using an epsilon factor helps the agent explore other states in the beginning
    - Using two networks (one for evaluation and one for testing) helps makes things easier. Use load state dict
    - Dropout and model complexity have large effects on convergence. Dropout might not be useful in scenarios like these'''

#imports
import math
import torch
from torch.nn import MSELoss
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
import random
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

#defining parameters
EPSILON = .9 #for greedy policy vs random
GAMMA = .9 #discount factor
LR = .0098 #learning rate
BATCH = 32 #batch size
REPLACE = 100 #updating target neural network # of learning iterations
MEMORY = 2000 #storing up a bunch of memory before training the neural networks. Then, randomly sampling from it
env = gym.make('CartPole-v0')
env = env.unwrapped
NACT = env.action_space.n
NSTATE = env.observation_space.shape[0]

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        layers = []
        layers.append(('input', nn.Linear(NSTATE, 75)))       # input layer
        layers.append(('input_activation', nn.ReLU(True)))
        layers.append(('out_lin', nn.Linear(75, NACT)))
        self.features = nn.Sequential(OrderedDict([*layers]))

    def forward(self, input):
        input = self.features(input)
        return input

class DQNet(object):
    def __init__(self):
        self.evaluation_net = Network()
        self.target_net = Network()
        self.optimizer = torch.optim.Adam(self.evaluation_net.parameters(), lr = LR)
        self.loss_fnc = nn.MSELoss()
        self.replace_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY, NSTATE * 2 + 2)) #stores state, next state, action, and reward

    def chooseAction(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        num = random.uniform(0,1)
        if (num < EPSILON): #greedy
            actions = self.evaluation_net.forward(state)
            action = torch.max(actions, 1)[1].data.numpy()[0]
        else: #random
            action = np.random.randint(0,2)
        return action

    def store_memory(self, state, reward, action, nextState):
        index = self.memory_counter % MEMORY
        insert = np.hstack((state, reward, action, nextState))
        self.memory[index, :] = insert
        self.memory_counter += 1

    def train_cust(self):
        #update the target net if necessary
        self.evaluation_net.train()
        if self.replace_counter % 100 == 0:
            self.target_net.load_state_dict(self.evaluation_net.state_dict())
        self.replace_counter += 1

        #sample from memory
        index = np.random.choice(MEMORY, BATCH)
        batch = self.memory[index, :]
        batch_prevState = torch.FloatTensor(batch[:, :NSTATE])
        batch_reward = torch.FloatTensor(batch[:, NSTATE: NSTATE + 1])
        batch_action = torch.LongTensor(batch[:, NSTATE + 1: NSTATE + 2])
        batch_newState = torch.FloatTensor(batch[:, -NSTATE:])
        #update the q value
        q = self.evaluation_net(batch_prevState).gather(1, batch_action)
        qnext = self.target_net(batch_newState).detach()
        qtar = batch_reward + GAMMA * qnext.max(1)[0].view(BATCH, 1)
        loss = self.loss_fnc(q, qtar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqnet = DQNet()

ANGLE = 1 #hoped that the environment would proplerly respond to us taking into account only 2 variables
POSITION = 1.2 #changed since the pole angle was good but kept moving translationally
exp = []

for i in range(175):
    state = env.reset()
    frames = 0
    done = False
    while not done:
        env.render()
        a = dqnet.chooseAction(state)
        newState, reward, done, info = env.step(a)

        #reward
        angleR = (.21 - abs(newState[2])) / (.21)
        positionR = (2.4 - abs(newState[0])) / (2.4)
        reward = angleR * ANGLE + positionR * POSITION
        dqnet.store_memory(state, reward, a, newState)
        frames += 1
        if dqnet.memory_counter > MEMORY:
            dqnet.train_cust()
            if done:
                print("Program exited after", frames, "frames.")
        if done:
            exp += [frames]
            break
        state = newState
plt.title("Duration over Iterations")
plt.plot(list(range(len(exp))), exp, 'b--')
plt.show()
env.close()
