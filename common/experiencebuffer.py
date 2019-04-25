import numpy as np

import torch


class ExpericenceBuffer:
    def __init__(self, states, actions, rewards, advantages):
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._advantages = advantages

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def advantages(self):
        return self._advantages

    def serialize(self, path):
        torch.save({'states': self._states,'actions': self._actions,'rewards': self._rewards,'advantages': self._advantages}, path)

    def deserialize(self, path):
        saved = torch.load(path)
        return ExpericenceBuffer(saved['states'], saved['actions'], saved['rewards'], saved['advantages'])

    @staticmethod
    def combine_experience(collectors):
        combined_states = np.concatenate(
            [np.array(c.states) for c in collectors])
        combined_actions = np.concatenate(
            [np.array(c.actions) for c in collectors])
        combined_rewards = np.concatenate(
            [np.array(c.rewards) for c in collectors])
        combined_advantages = np.concatenate(
            [np.array(c.advantages) for c in collectors])
        return ExpericenceBuffer(combined_states, combined_actions, combined_rewards, combined_advantages)
