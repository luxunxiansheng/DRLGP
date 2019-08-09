from branch import Branch
from node   import Node 
from common.move  import Move


class Tree(object):
    def __init__(self):
        self._root_node=Node

    
    @property
    def root_node(self):
        return self._root_node
    

    def add_node_to_branch(self, game_state, temperature,estimated_branch_priors, estimated_state_value, parent_branch):
        new_node = Node(game_state, estimated_state_value, parent_branch,temperature)
        parent_branch.child_node = new_node

        return new_node          
    
   
    def add_branch_to_node(self,parent_node,point,prior):
        new_branch= Branch(parent_node,Move(point),prior)
        parent_node[point]=new_branch

        return new_branch

    
    def add_node(self, game_state, estimated_branch_priors, estimated_state_value, parent_branch=None):
        new_node = Node(game_state, estimated_state_value, parent_branch, self._temperature)

        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[point] = Branch(new_node, Move(point), p)

        new_node.children_branch = chiildren_branch

        if parent_branch is not None:
            parent_branch.child_node = new_node

        return new_node    


 

