import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from dqn_agent import Agent

def dqn(agent, n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.1, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting values of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores for averaging scores
    esp = eps_start                    # initialize epsilon
    
    # Run episodes
    for i_episode in range(1, n_episodes+1): 
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        # Run time steps
        for t in range(max_t):
            action = agent.act(state, esp)                 # select action based on the greedy policy
            # take the action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward of chosen action
            done = env_info.local_done[0]                  # get the done status 
            
            # update the agent
            agent.step(state, action, reward, next_state, done)
            
            # accumulate the score
            score += reward
            if done:
                break

            # proceed to the next state
            state = next_state
                                    
        scores_window.append(score)         # add score to the queue
        scores.append(score)                # add score to the list
        esp = max(eps_end, eps_decay*esp)   # decrease epsilon

        # print the average score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        # print the average score every 100 episodes
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:  # stop if the average score is greater than 13.0
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # save the weights
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=True)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

scores = dqn(agent)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('scores')
plt.xlabel('Episode #')
plt.show()