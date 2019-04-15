from common.player import Player


class Agent(Player):
    
    def select_move(self, game_state):
        raise NotImplementedError()
