from unityagents import UnityEnvironment
from dqn_agent import Agent
import torch

def main():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    maxT = 2000000
    t = 0
    # load the weights from file
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    last_score = 0
    while t < maxT:
        action = agent.act(state)                      # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward                                # update the score
        if score != last_score:
            print('\rscore {}'.format(score))
            last_score = score
        state = next_state                             # roll over the state to next time step
        if done:
            print("done")
            break
        t +=1

    env.close()
    print("Score: {}".format(score))           
    return

if __name__ == "__main__":
    main()