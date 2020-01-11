import numpy as np

from common.move import Move


class MCTSNode(object):
    def __init__(self, game_state, prior_p, parent=None):
        self._game_state = game_state
        self._parent = parent
        self._children = {}
        self._num_visits = 0

        self._Q = 0
        self._P = prior_p
        self._u = 0

    def select(self, c_puct):
        child = max(self._children.items(),
                    key=lambda point_node: point_node[1].get_value(c_puct))
        return child[1]

    def update_recursively(self, root_node, leaf_value):
        if self != root_node and self._parent:
            self._parent.update_recursively(root_node, -leaf_value)

        self._num_visits += 1
        self._Q += 1.0*(leaf_value-self._Q)/self._num_visits

    def is_leaf(self):
        return self._children == {}

    @property
    def num_visits(self):
        return self._num_visits

    @property
    def game_state(self):
        return self._game_state

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    def get_child(self, point):
        return self._children.get(point)

    def add_child(self, game, new_point, prior):
        new_game_state = game.look_ahead_next_move(
            self._game_state, Move(new_point))
        new_node = MCTSNode(new_game_state, prior, self)
        self._children[new_point] = new_node
        return new_node

    def get_value(self, c_puct):
        """
        Same as in alphazero
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._num_visits) / (1 + self._num_visits))
        return self._Q + self._u
