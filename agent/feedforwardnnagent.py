import numpy as np 
import torch 

import pysnooper

from common.gamestate              import GameState
from common.move                   import Move
from common.player                 import Player
from common.encoder                import Encoder
from common.board                  import Board
from models.feedfowrdnerualnetwork import FeedForwardNeuralNetwork


class FeedForwardNeuralNetworkAgent(Player):
    def __init__(self,id,name,mark,encoder,board_size,state_dict):
        super().__init__(id,name,mark)
        self._model= FeedForwardNeuralNetwork()
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._encoder = encoder
        self._num_points = board_size*board_size

    #@pysnooper.snoop()
    def select_move(self,game,game_state):
        point_probs= self._predict(game_state).detach().numpy()
        point_probs = point_probs **3
        eps= 1e-6
        point_probs = np.clip(point_probs,eps,1-eps)
        point_probs = point_probs / np.sum(point_probs)
        possible_points=np.arange(self._num_points)
        ranked_points=np.random.choice(possible_points,size=self._num_points,replace=False,p=point_probs[0])
        for point_index in ranked_points:
            point = self._encoder.decode_point_index(point_index)
            if game_state.board.is_valid_point(point):
                return Move(point)


    def _predict(self,game_state):
        board_matrix = self._encoder.encode(game_state)
        board_matrix=board_matrix.reshape([1,self._num_points])

        return self._model(torch.tensor(board_matrix,dtype=torch.float))


        

