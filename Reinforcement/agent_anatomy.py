import random

class Environment:
    def __init__(self):
        self.steps_left = 10 # The number of time steps the agent is allowed to take to interact with the environment

    # The get_observation() method is supposed to return the current environment's observation to the agent

    def get_observation(self):
        return [0.0, 0.0, 0.0]

    # The get_actions() method allows the agent to query the set of actions it can execute
    
    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0
    
    # The action() method does two things : 
    # 1) Handles the agent's action 
    # 2) Returns the reward for this action
    
    def action(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -=1
        return random.random()

    
class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env):
        current_obs = env.get_observation() # Observe the environment
        actions = env.get_actions() # Get the actions to be done
        reward = env.action(random.choice(actions)) # Get rewards for the actions performed
        self.total_reward += reward


if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)
    print("Total reward got : {:.9f}".format(agent.total_reward))

    
    




    

    