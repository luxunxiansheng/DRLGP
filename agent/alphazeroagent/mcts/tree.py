from branch import Branch
from node   import Node 
from common.move  import Move


class Tree(object):
    
    def __init__(self):
        self._root_node = None
        self._working_node = None

    @property
    def root_node(self):
        return self._root_node

    @root_node.setter
    def root_node(self, value):
        self._root_node = value    
    
    @property
    def working_node(self):    
        return self._working_node

    @working_node.setter
    def working_node(self, value):
        self._working_node = value

    def reset(self):
        self._root_node= None
        self._working_node =None    

    def go_down(self, move):
        branch = self._working_node.get_child_branch(move.point)
        self._working_node = branch.child_node

    
   
   
    
    


 

