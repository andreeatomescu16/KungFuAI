from env import *

def evaluate(agent, env, n_episodes = 1):
    episodes_rewards = []
    for  _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            state, reward, done, info, _ = env.step(action[0])
            total_reward += reward
            if done:
                break
        
        episodes_rewards.append(total_reward)

    return episodes_rewards