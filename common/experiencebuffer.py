import numpy as np


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

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self._states)
        h5file['experience'].create_dataset('actions', data=self._actions)
        h5file['experience'].create_dataset('rewards', data=self._rewards)
        h5file['experience'].create_dataset(
            'advantages', data=self._advantages)

    def deserialize(self, h5file):
        return ExpericenceBuffer(states=np.array(h5file['experience']['states']),
                                 actions=np.array(h5file['experience']['actions']),
                                 rewards=np.array(h5file['experience']['rewards']),
                                 advantages=np.array(h5file['experience']['advantages']))

    @staticmethod
    def combine_experience(collectors):
        combined_states = np.concatenate([np.array(c.states) for c in collectors])
        combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
        combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
        combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])
        return ExpericenceBuffer(combined_states, combined_actions, combined_rewards, combined_advantages)
