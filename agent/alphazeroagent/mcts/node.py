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

from  agent.alphazeroagent.mcts.branch  import Branch
from  common.move  import  Move

class Node(object):
    def __init__(self, game_state, game_state_value, parent_branch, temperature=0.8):
        self._game_state = game_state
        
        self._game_state_value = game_state_value
        self._total_visit_counts = 1

        self._parent_branch = parent_branch
        self._children_branch = {}

        self._temperature = temperature


    def add_branch(self,point,prior):
       
        self._children_branch[point]=Branch(self,Move(point),prior)

    def does_branch_exist(self,point):
        return point in self.children_branch
  
    def get_child_branch(self, point):
        return self.children_branch[point]
        
    def expected_value_of_branch(self, point):
        return self._children_branch[point].expected_value
        
    def prior_of_branch(self, point):
        return self._children_branch[point].prior
        
    def visit_counts_of_branch(self, point):
        return self._children_branch[point].visit_counts
        
    def record_visit(self, point, value):
        self._total_visit_counts += 1
        self._children_branch[point].visit_counts += 1
        self._children_branch[point].total_value += value

    def select_branch(self, randomly=False, is_selfplay=True):
        Qs = [self.expected_value_of_branch(point) for point in self.children_branch]
        Ps = [self.prior_of_branch(point) for point in self.children_branch]
        Ns = [self.visit_counts_of_branch(point) for point in self.children_branch]

        if randomly and is_selfplay:
            noises = np.random.dirichlet([0.03] * len(self.children_branch))
            Ps = [0.75*p+0.25*noise for p, noise in zip(Ps, noises)]

        scores = [(q + self._temperature * p * np.sqrt(self._total_visit_counts) / (n + 1)).item() for q, p, n in zip(Qs, Ps, Ns)]
        best_point_index = np.argmax(scores)

        points = list(self.children_branch)
        return self._children_branch[points[best_point_index]]

    @property
    def game_state(self):
        return self._game_state

    @property
    def parent_branch(self):
        return self._parent_branch

    @property
    def game_state_value(self):
        return self._game_state_value

    @property
    def temperature(self):
        return self._temperature

    @property
    def children_branch(self):
        return self._children_branch.keys()

    @children_branch.setter
    def children_branch(self, value):
        self._children_branch = value

    def is_leaf(self):
        return self._children_branch 