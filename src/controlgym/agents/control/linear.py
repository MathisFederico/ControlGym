import numpy as np

from controlgym.agents.agent import Agent


class LinearControlAgent(Agent):
    def __init__(self, K, obs_to_state=lambda x: x):
        self.K = np.array(K)
        self.obs_to_state = obs_to_state

    def act(self, observation):
        state = self.obs_to_state(observation)
        return np.dot(self.K, state)
