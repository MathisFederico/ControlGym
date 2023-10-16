from gym.spaces import Space

from controlgym.agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()
