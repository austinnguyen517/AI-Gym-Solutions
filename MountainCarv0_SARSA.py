'''Austin Nguyen ~ 6/21/2019 ~ Problem #2
MountainCar v0  ~ SARSA (State-Action-Reward-State-Action)
Results: Good performance. Within maybe 50-75 episodes, the agent learns relatively well. However, it can be inconsistent probably due to a high LR
Changes to Consider: Consider implementing code that is not so susceptible to initial state of network'''

'''
    Intuition:
        - An on policy version of DQN.
        - Learns Q value not based on greedy updates using optimal action but rather the current policy
        - If the agent provides an episode that is the best so far (meaning it got to the top quickly), train the network on the data from the last part of that episode.
            - Basically, tell it to do the same exact thing again! It probably did something right towards the end if it got to the top
        - If the agent takes more than 500 frames to solve the problem, just start a new episode ~ don't overcrowd the experience replay with bad data

    Takeaways:
        - Implementing a reward function that is not PERFECTLY intuitive is okay. Sometimes, relying on the discount factor is enough
        - Rewarding the network for providing a good episode is enough to train it. Training it to replicate that episode is a good method sometimes
        - Taking the agent away and starting fresh when the episode is not working out will sometimes help

'''

#imports
import torch
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import gym
import torch.nn.functional as F
import matplotlib.pyplot as plt

#training parameters (all caps)
LR = .0012
EPSILON = .9  #greedy vs probabilistic
BATCH_SIZE = 32
MAX_MEMORY = 1000
UPDATE = 100
GAMMA = .95 #discount factor

#define the neural network class and what you want in it
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.loss_fnc = nn.MSELoss()
        self.inputN = 2
        self.outputN = 3
        self.features = nn.Sequential(
            nn.Linear(self.inputN, 150),
            nn.ReLU(True),
            nn.Linear(150, self.outputN)
        )

    def forward(self, x):
        return self.features(x)

#define the SARSA net
class SARSA(object):
    def __init__(self):
        self.predictNet = Network()
        self.updateNet = Network()
        self.loss_fnc = nn.MSELoss()
        self.optimizer = optim.Adam(self.predictNet.parameters(), lr = LR)
        self.memory = np.zeros((MAX_MEMORY, 7))
        self.memoryCount = 0
        self.updateCount = 0

    def nextAction(self, state):
        value = np.random.uniform()
        if value < EPSILON: #greedy
            output = self.predictNet.forward(state)
            action = torch.argmax(output).data.numpy()
        else:
            action = np.random.randint(0, 3)
        return action

    def update(self):
        self.updateNet.load_state_dict(self.predictNet.state_dict())

    def storeMemory(self, input):
        index = self.memoryCount % MAX_MEMORY
        self.memory[index, :] = input
        self.memoryCount += 1

    def train(self, all = False, frames = 200):
        self.predictNet.train()
        if self.updateCount % UPDATE == 0:
            self.update()
        self.updateCount += 1

        if all:
            index = np.array(list(range(frames)))
            #index = np.array(list(range(MAX_MEMORY - 250, MAX_MEMORY)))
        else:
            index = np.random.choice(MAX_MEMORY, BATCH_SIZE)
        batch = self.memory[index, :]
        prevState = torch.FloatTensor(batch[:, [0,1]])
        prevAct = torch.LongTensor(batch[:, 2:3])
        rewards = torch.FloatTensor(batch[:, 3:4])
        if all:
            rewards += 2
        nextState = torch.FloatTensor(batch[:, [4,5]])
        nextAct = torch.LongTensor(batch[:, 6:7])
        q = self.predictNet(prevState).gather(1,prevAct)
        qtarget = rewards + GAMMA * self.predictNet(nextState).detach().gather(1,nextAct)
        loss = self.loss_fnc(q, qtarget)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


env = gym.make('MountainCar-v0')
env = env.unwrapped
net = SARSA()
exp = []

for i in range(100):
    print("Beginning episode number: ", i)
    print("")
    state = env.reset()
    frames = 0
    done = False
    best = None
    while not done:
        env.render()
        action = net.nextAction(torch.FloatTensor(state))

        nextState, reward, done, info = env.step(action)
        nextAct = net.nextAction(torch.FloatTensor(nextState))
        reward = nextState[0] + .6
        if nextState[0] >= .5:
            reward += 10
        frames += 1
        net.storeMemory(np.hstack((state, action, reward, nextState, nextAct)))
        if net.memoryCount > MAX_MEMORY:
            net.train()
        if frames > 500:
            break
        if done:
            if best == None or frames < 200 or frames < best:
                net.train(True, min(frames, 200))
                best = frames
            print("Episode ended after: ", frames, " frames")
            exp += [frames]
            break
        state = nextState
        action = nextAct

plt.plot(list(range(len(exp))), exp, 'b--')
plt.title("Duration over Iterations")
plt.show()
env.close()
