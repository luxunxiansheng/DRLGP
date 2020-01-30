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

import copy
from collections import deque

import numpy as np


class ExpericenceBuffer:
    def __init__(self, compacity=10000):
        self._data = deque(maxlen=compacity)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = copy.deepcopy(data)

    def combine_experience(self, collectors):
        combined_states = np.concatenate(
            [np.array(c.states) for c in collectors])
        combined_rewards = np.concatenate(
            [np.array(c.rewards) for c in collectors])
        combined_visit_counts = np.concatenate(
            [np.array(c.visit_counts) for c in collectors])

        zipped_data = zip(combined_states, combined_rewards,
                          combined_visit_counts)
        self._data.extend(zipped_data)

    def size(self):
        return len(self._data)

    def merge(self, other):
        self._data += other.data
