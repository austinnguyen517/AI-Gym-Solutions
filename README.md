# AI-Gym-Solutions
Solving Gym AI RL problems using variety of RL algorithms. 

### Within each file:
  - Date of creation
  - Results of code 
  - Intuition behind implementation
  - Takeaways from each problem

### DQN ~ CartPole
  Using network to find Q values (off policy) associated with state action pairs. Simple implementation using experience replay and random   sampling.

### SARSA ~ MountainCarDiscrete
  Network to find Q values using on policy version of DQN. Uses experience replay, random sampling, and rewards entire successful episodes

### REINFORCE ~ MountainCarContinuous
  Using policy gradients on neural networks that output the mean and variance of a gausssian distribution (in continuous action space)
  
### Q Actor Critic ~ MountainCarContinuous
  Using actor critic method with the critic being the output of a state-action Q network
