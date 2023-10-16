from hebg.behavior import Behavior

from controlgym.agents.agent import Agent


class HEBGAgent(Agent):
    def __init__(self, he_behavior: Behavior) -> None:
        super().__init__()
        self.he_behavior = he_behavior

    def act(self, observation):
        return self.he_behavior(observation)
