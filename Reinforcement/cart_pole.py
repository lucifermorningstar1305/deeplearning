import gym
from random_action_wrapper import RandomActionWrapper
# env = gym.make('CartPole-v0')

# obs = env.reset()
# print(obs)
# print(env.action_space)
# print(env.observation_space)
# # while True:
# #     env.render()
# # env.close()
# print(env.step(0))

if __name__ == '__main__':
    env = RandomActionWrapper(gym.make('CartPole-v0'))
    # env = gym.wrappers.Monitor(env, "recording", force=True)
    obs = env.reset()
    total_rewards = 0.0
    total_steps = 0
    while True:
        env.render()
        action = env.action_space.sample()
        obs, rewards, done, _= env.step(action)
        total_rewards += rewards
        total_steps += 1
        if done:
            print("Total Episode {},  rewards: {:.4f}".format(total_steps, total_rewards))
            break
    env.close()