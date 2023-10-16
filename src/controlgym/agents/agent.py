from abc import abstractmethod


class Agent:
    @abstractmethod
    def act(self, observation):
        """Agent action from given observation"""

    def reset(self):
        """Reset agent for new episode"""

