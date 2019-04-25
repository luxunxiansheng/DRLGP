class ExperienceCollector():
    def __init__(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_esimated_values = []

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


    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_esimated_values = []
    
    def record_decision(self, state, action, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action) 
        self._current_episode_esimated_values.append(estimated_value)

    def compelte_episode(self,reward):
        num_states= len(self._current_episode_states)
        self._states += self._current_episode_states
        self._actions+= self._current_episode_actions
        self._rewards+= [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - self._current_episode_esimated_values[i]
            self._advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_esimated_values = []
         

        