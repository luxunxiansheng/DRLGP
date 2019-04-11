from common.player import Player

class Agent(Player):
    def __init__(self,name):
        Player.__init__(self,name)

    def select_move(self, game_state):
        raise NotImplementedError()

