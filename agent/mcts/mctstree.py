from common.move import Move
from agent.mcts.mctsnode import MCTSNode

class MCTSTree(object):
    def __init__(self):
        self._working_node = None

    @property
    def working_node(self):
        return self._working_node

    @working_node.setter
    def working_node(self, value):
        self._working_node = value

    def reset(self):
        self._working_node = None

    def go_down(self, game, move):
        if self._working_node is not None:
            if move.point in self._working_node.children:
                child = self.working_node.children.pop(move.point)
                child.parent = None
            else:
                if not self._working_node.is_terminal(game):
                    new_game_state = game.look_ahead_next_move(
                        self._working_node.game_state, move)
                    child = MCTSNode(new_game_state, 1.0, None)
                else:
                    child = None
            self._working_node = child
