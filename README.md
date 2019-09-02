# AI-Gym-Solutions
Solving Gym AI RL problems using variety of RL algorithms. 

### Within each file:
  - Date of creation
  - Results of code 
  - Intuition behind implementation
  - Takeaways from each problem

## DQN ~ CartPole
  Using network to find Q values (off policy) associated with state action pairs. Simple implementation using experience replay and random   sampling.
  
####      Results:
  - Results: Very solid duration for each iteration. Translational motion was harder to temper than angular motion.

  - Takeaways:
    - Random individual sampling from experience replay -> more stable results than just than sampling in contiguous chunks
    - Using an epsilon factor helps the agent explore other states in the beginning
    - Using two networks (one for evaluation and one for testing) helps makes things easier. Use load state dict
  
## SARSA ~ MountainCarDiscrete
  Network to find Q values using on policy version of DQN. Uses experience replay, random sampling, and rewards entire successful episodes
  
####      Results:
  - Results: Good performance. Within maybe 50-75 episodes, the agent learns relatively well. However, it can be inconsistent 
    
  - Takeaways:
      - Implementing a reward function that is not PERFECTLY intuitive is okay. Relying on the discount factor is enough
      - Rewarding the network for a good episode works. Training it to replicate that episode is a good method sometimes
      - Taking the agent away and starting fresh when the episode is not working out will sometimes help

## REINFORCE ~ MountainCarContinuous
  Using policy gradients on neural networks that output the mean and variance of a gausssian distribution (in continuous action space)
  
####       Results:

   - Results: UNABLE TO LEARN. The implementation does not work especially because the reward system makes it extraordinarily difficult. The reinforce algorithm does not directly translate to continuous action spaces, and makes it extremely hard to learn. Next time, try an actor critic method and see what happens.
   - Takeaways:
        - REINFORCE relies on policy gradients. This suggests contiguous points of data for training
        - This makes it not ideal for environments like this where the reward system encourages the agent to do nothing at all
  
## Q Actor Critic ~ MountainCarContinuous
  Using actor critic method with the critic being the output of a state-action Q network
