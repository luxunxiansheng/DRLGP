class ExpericenceBuffer():
    def __init__(self, states, actions, rewards,advantages):
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._advantages= advantages

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self._states)
        h5file['experience'].create_dataset('actions', data=self._actions)
        h5file['experience'].create_dataset('rewards', data=self._rewards)
        h5file['experience'].create_dataset('advantages',data=self._advantages)


    