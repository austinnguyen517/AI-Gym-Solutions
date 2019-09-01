'''Austin Nguyen ~ July 15
   MountainCarv0-Continuous ~ Q Actor Critic

   Results:
        -

   Changes to consider: '''

'''
Intuition:
    - We use a different version of the policy gradient. The critic estimates the value function.
    - The actor updates the policy distribution in direction suggested by critic
    - We use two different networks. One that outputs a policy distribution and another that outputs Q values of state action pairs
    - We first update policy parameters by using gradient of log probabilities and Q values
    - Then we calculate TD errors and use that as a "weight" to update Q function parameters
    - Basically, after updating Q values in the Q network, we want to increase the probability that we make decisions that have higher Q values
    - However, since mountaincar does not have a very generous reward system, consider restarting training if it doesn't reach the top at a certain points OR using TRPO


    Temporal Difference (TD):
        - R(t+1) + (gamma)V(S(t+1)) - V(S(t))
        - Difference between the actual reward and the expected reward
        - We are going to use this as our "weight" to update the Q function
        - This is sometimes used as its own learning method (look of TD learning as opposed to Monte Carlo learning)
            - Monte Carlo assumes that the Q value is the reward of the current state.

    Trust Region Policy Optimization (TRPO)
        - The math was too hard to implement, but the idea is to update policies relative to past policies. Then, using a lower bound on the function we are trying to "maximize,"
          we update the current policy using a trust region...searching that area to find a method that guarantees to optimize performance
        - KL divergence is the expected difference in log probability of two policies or distributions. TRPO uses this when parameterizing its trust region
        - We basically replace the log probabilities from policy gradients with ratios of probabilities from new and old policies
        - Much harder to implement than you think!
Takeaways:
    -
'''

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import gym
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as tdist
from torch.autograd import Variable
import sklearn
import sklearn.preprocessing
from multiprocessing import Process

LRQ = .0007
LRP = .00002
Qstep = 400
Pstep = 400
Qgamma = .9
Pgamma = .9
GAMMA = .9
MEMORY = 1200
HIDWIDTHQ = 400
HIDWIDTHPOLICY = 40
NUMITERATIONS = 2000
MAXFRAMES = 400
RESET = 5
VISUAL = False
BATCH = 16
REWARDLENGTH = 10

class Net(nn.Module):
    def __init__(self, n_in, n_out, hidWidth, prob):
        super(Net, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidWidth = hidWidth
        self.prob = prob
        self.features = nn.Sequential(
            nn.Linear(self.n_in, self.hidWidth),
            nn.ELU(),
            nn.Linear(self.hidWidth, self.hidWidth),
            nn.ELU(),
            nn.Linear(self.hidWidth, self.n_out)
        )
    def forward(self, input):
        if not self.prob:
            return self.features(input)
        else:
            out = self.features(input)
            split = int(self.n_out/2)
            mean = torch.tanh(out.narrow(0,0,split))
            var = torch.sigmoid((out.narrow(0, split, split))) + 1e-5
            return mean, var
    def init_weight_orth(self):
        # inits the NN with orthogonal weights
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
        self.features.apply(init_weights)

def normal(x, mu, var):
    expTerm = (-1 * (Variable(x) - mu).pow(2)/(2 * var)).exp()
    coefTerm = 1/(2 * var * (3.14159265)).sqrt()
    return coefTerm * expTerm


class QAC():
    def __init__(self, state_n, action_n):
        self.memory = []
        self.QNet = Net(state_n + action_n, 1, HIDWIDTHQ, False)
        self.policy = Net(state_n, 2*action_n, HIDWIDTHPOLICY, True) #multiply by 2 to give variances
        self.optimizerQ = optim.SGD(self.QNet.parameters(), lr = LRQ, momentum = .8)
        self.optimizerPolicy = optim.SGD(self.policy.parameters(), lr = LRP, momentum = .8)
        self.stepperQ = optim.lr_scheduler.StepLR(self.optimizerQ, Qstep, gamma = Qgamma)
        self.stepperP = optim.lr_scheduler.StepLR(self.optimizerPolicy, Pstep, gamma = Pgamma)
        self.memoryCount = 0
        self.hasSucceeded = False

#implement experience replay that stores rewards, state, action, probability state/action, next state, and next action
    def chooseAction(self, state):
        mean, var = self.policy(torch.FloatTensor(scale_state(state)))
        eps = torch.randn(mean.size())
        action = torch.clamp((mean + var.sqrt() * Variable(eps)).data, -1, 1)
        logprob = normal(action, mean, var).log()
        return action, logprob


    def storeMemory(self, s, a,logprob, r, s1, a1):
        if self.memoryCount >= MEMORY:
            self.memory = self.memory[1:]
        self.memory += [[s,a,logprob,r,s1,a1]]
        self.memoryCount += 1

    def init_weight_orth(self):
        self.policy.init_weight_orth()
        self.QNet.init_weight_orth()


    def train(self, last = False):
        #idx = np.random.choice(MEMORY, 1).tolist()[0]
        idx = np.random.choice(MEMORY, BATCH).tolist()
        self.hasSucceeded = self.hasSucceeded or last
        if last:
            idx += [self.memoryCount - 1 if self.memoryCount < MEMORY else MEMORY - 1]
        for i in idx:
            batch = self.memory[i] #assume batch is 1
            s = torch.FloatTensor(scale_state(batch[0]))
            a = batch[1]
            logprob = batch[2]
            r = batch[3]
            s1 = torch.FloatTensor(scale_state(batch[4]))
            a1 = batch[5]

            Q = self.QNet((torch.FloatTensor(torch.cat((s, a), 0))))
            Qclone = Q.clone().detach()
            policyGrad = -1 * logprob * Qclone

            Q1 = self.QNet((torch.FloatTensor(torch.cat((s1, a1), 0)))).detach()
            TD = r + (GAMMA*(Q1)) - Qclone
            QGrad = -1 * TD * Q

            policyGrad.backward(retain_graph = True)
            QGrad.backward()

        self.optimizerPolicy.zero_grad()
        self.optimizerPolicy.step()

        self.optimizerQ.zero_grad()
        self.optimizerQ.step()

        if self.hasSucceeded:
            self.stepperQ.step()
            self.stepperP.step()

env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
network = QAC(2, 1)
network.init_weight_orth()
rewards = []
success = False


#function to normalize states
def scale_state(state):
    result = [0,0]              #requires input shape=(2,)
    result[0] = (state[0] + .3) / .9
    result[1] = state[1] / .07
    return result

for i in range(NUMITERATIONS):
    state = env.reset()
    if (i % RESET == 0 and not success):
        print("")
        print("")
        print("############################## Reseting network... ###############################")
        print("")
        print("")
        network = QAC(2,1) #reset the network
        network.init_weight_orth()
        rewards = []
    frames = 0
    done = False
    aggreward = 0
    action, logprob = network.chooseAction(torch.FloatTensor(state))

    while not done:
        if VISUAL:
            env.render()
        newState, reward, done, info = env.step(action)
        aggreward += reward
        newAction, newlogprob = network.chooseAction(torch.FloatTensor(newState))
        network.storeMemory(state, action, logprob, reward, newState, newAction)
        frames += 1
        if frames >= MAXFRAMES:
            done = True
        if network.memoryCount >= MEMORY: #network.memoryCount >= MEMORY: #network.memoryCount >= MEMORY:
            network.train()
        if done:
            #if frames < MAXFRAMES or newState[0] >= .40 or abs(newState[1]) >= .025:
            if frames < MAXFRAMES:
                print("*************** SUCCESS!!!! ********************")
                success = True
                network.train(True)
            print("Episode ", i + 1," ended after: ", frames, " frames with aggregate reward of: ", aggreward)
            print("")
            rewards += [aggreward]
            break
        state = newState
        action = newAction
        logprob = newlogprob
plt.plot(list(range(len(rewards))), rewards, 'b--')
plt.title("Rewards over Iterations")
plt.show()
env.close()
