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

class ExperienceCollector:
    def __init__(self):
        self._states = []
        self._visit_counts = []
        self._rewards = []
        

    def reset(self):
        self._states = []
        self._visit_counts = []
        self._rewards = []

    def record_decision(self, state, visit_counts):
        
        augmented_state =[]
        augmented_visit_counts=[]
        
        reshaped_visit_counts= visit_counts.reshape(state.shape[1],state.shape[2]) 

        for i in [1,2,3,4]:
            
            counterclockwise_rotated_state= np.array([np.rot90(plane,i) for plane in state])
            counterclockwise_rotated_visit_counts= np.rot90(reshaped_visit_counts,i)
            augmented_state.append(counterclockwise_rotated_state)
            augmented_visit_counts.append(counterclockwise_rotated_visit_counts.flatten())


            horizontally_flipped_state= np.array([np.fliplr(plane) for plane in counterclockwise_rotated_state]) 
            horizontally_flipped_visit_counts= np.fliplr(counterclockwise_rotated_visit_counts)
            augmented_state.append(horizontally_flipped_state)
            augmented_visit_counts.append(horizontally_flipped_visit_counts.flatten())
        
        self._states.extend(augmented_state)
        self._visit_counts.extend(augmented_visit_counts)  
        

    def complete_episode(self, reward):
        num_states = len(self._states)
        self._rewards += [reward for _ in range(num_states)]
        self.reset()

    @property
    def visit_counts(self):
        return self._visit_counts

    @property
    def rewards(self):
        return self._rewards

    @property
    def states(self):
        return self._states
