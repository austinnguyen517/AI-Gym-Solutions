'''Austin Nguyen ~ 6/25/2019
MountainCarContinuous-v0 ~ REINFORCE
Results:
    UNABLE TO LEARN.
    The implementation does not work especially because the reward system makes it extraordinarily difficult.
    The reinforce algorithm does not directly translate to continuous action spaces, and makes it extremely hard to learn. Next time, try an actor
    critic method and see what happens.
Changes to consider: Change the reward system? Consider not training on entire episodes. Let exploration be more dynamic and useful
'''

'''
Intuition:
    - Intuition: Use a neural network that outputs a Gaussian probability distribution based on mean and std deviation
    - Use this policy to input into log probabilistic loss
    - Propagate this loss as it automatically takes the gradient for us WRT network parameters
    - Used same idea as discrete mountaincar where entire good sequences of good actions were rewarded regardless of sampling
Takeaways:
    - I know how the REINFORCE algorithm works. However, for this specific case, it just was unable to work 
'''
#imports
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import gym
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable

#defining parameters
LR = .00001
EPSILON = .7 #greedy vs random
EPSINC = .04
GAMMA = .9
POSWEIGHT = 10
VELWEIGHT = 20

def normal(x, mu, var):
    expTerm = (-1 * (Variable(x) - mu).pow(2)/(2 * var)).exp()
    coefTerm = 1/(2 * var * (3.14159265)).sqrt()
    return coefTerm * expTerm
#defining the network
class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.inputN = 2
        self.outputN = 2 #mean and stddev
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 1)
        self.linear2_ = nn.Linear(100,1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        var = self.linear2_(x)

        return mu, var

#defining overall REINFORCE class
class Reinforce():
    def __init__(self):
        self.network = Net()
        self.epstep = 0
        self.epsilon = EPSILON
        self.optimizer = optim.Adam(self.network.parameters(), lr = LR)

    #get action
    def normalizeState(self, state):
        posNorm = (state[0] + 1.2)/1.8
        velNorm = (state[1] + .07)/.14
        return [posNorm, velNorm]

    def getAction(self, state):
        mu, var = self.network.forward(Variable(state))
        random = np.random.uniform()
        if random > self.epsilon: #random
            mu += np.random.randint(-1, 1)
            var += np.random.randint(-1,1)
        var = F.softplus(var)
        eps = torch.randn(mu.size())
        action = (mu + var.sqrt() * Variable(eps)).data #basically, take the outputted mean and randomly determine how much of the variance we will add on to it with a normal distribution
        prob = normal(action, mu, var)

        log_prob = prob.log()
        '''random = np.random.uniform()
        mean = output[0]
        std = torch.abs(output[1])
        if random > self.epsilon: #random
            mean = mean + np.random.uniform(-5, 5)
            std = std + np.random.uniform(0,10)
        distribution = tdist.Normal(mean, std)
        action = distribution.sample(torch.Size([1,1]))
        logprob = (distribution.log_prob(action[0][0])).clone()'''
        return action, log_prob

    #train method
    def train(self, logprobs, rewards):
        self.network.train()
        discounted_rewards = []
        self.epsilon += (.9 - self.epsilon) * EPSINC
        print("New epsilon: ", self.epsilon)

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = []
        for log_prob, g in zip(logprobs, discounted_rewards):
            loss.append(-log_prob * g)
        loss = torch.sum(torch.stack(loss))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

#define the environment
env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
network = Reinforce()
exp = []
losses = []

#main loop
for episode in range(200):
    #for many episodes
    state = env.reset()
    frames = 0
    done = False
    logprobs = []
    rewards = []

    while not done:
        env.render()
        state = network.normalizeState(state)
        action, logprob = network.getAction(torch.FloatTensor(state))
        newState, reward, done, info = env.step(action)
        reward = POSWEIGHT * ((newState[0] + 1.2)/1.8) + VELWEIGHT * ((abs(newState[1]))/.07)

        if newState[0] >= .5:
            reward += 100
        rewards.append(reward)
        logprobs.append(logprob)
        frames += 1
        if frames > 599:
            done = True
        if done:
            loss = network.train(logprobs, rewards)
            print("Episode ended after: ", frames, " frames")
            losses += [loss.detach().numpy()]
            print("Loss: ", losses[-1])
            exp += [frames]
            break
        state = newState

plt.plot(list(range(len(exp))), exp, 'b--')
plt.title("Duration over Iterations")
plt.show()
plt.plot(list(range(len(losses))), losses, 'b--')
plt.title("Loss over Iterations")
plt.show()
env.close()
