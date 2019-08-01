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
 #/  



class Branch:
    def __init__(self, parent_node,move,prior):
        self._move = move
        self._total_value = 0.0
        
        self._prior = prior
        self._visit_counts = 0

        self._parent_node  = parent_node
        self._child_node = None

    @property
    def move(self):
        return self._move

    @property
    def prior(self):
        return self._prior

    @property
    def total_value(self):
        return self._total_value

    @total_value.setter
    def total_value(self, value):
        self._total_value = value

    @property
    def visit_counts(self):
        return self._visit_counts

    @visit_counts.setter
    def visit_counts(self, value):
        self._visit_counts = value

    @property
    def parent_node(self):
        return self._parent_node

    @property
    def child_node(self):
        return self._child_node    
   
    @child_node.setter
    def child_node(self, node):
        self._child_node = node