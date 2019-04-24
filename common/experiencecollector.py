class ExperienceCollector():
    def __init__(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_esimated_values = []
        
    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_esimated_values = []
    
    def record_decision(self, state, action, estimated_value=0):
        
        