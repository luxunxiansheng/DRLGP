# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /
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
