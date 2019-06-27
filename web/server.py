import os
import sys
sys.path.insert(0, os.path.abspath('.'))


import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from models.connect5network import Connect5Network

from agent.alphazeroagent import (AlphaZeroAgent, AlphaZeroExpericenceBuffer,
                                  AlphaZeroExperienceCollector,
                                  MultiplePlaneEncoder)
from agent.humanplayer import HumanPlayer
from common.board import Board
from common.move import Move
from common.player import Player
from common.point import Point
from game.connect5game import Connect5Game


def coords_from_point(point):
    return '%s%d' % (Board.get_column_indicator(point.col-1), point.row)


def get_web_app(bot_map):
    """Create a flask application for serving bot moves.
    Returns: Flask application instance
    """
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, 'static')
    app = Flask(__name__, static_folder=static_path, static_url_path='/static')

    @app.route('/select-move/<bot_name>', methods=['POST'])
    def select_move(bot_name):
        content = request.json
        jboard_postion = HumanPlayer.point_from_coords(content['move'][0])
        move = Move(Point(jboard_postion.col,board_size+1-jboard_postion.row))
        game.apply_move(move)

        bot_agent = bot_map[bot_name]
        bot_move = bot_agent.select_move(game, game.working_game_state)
        game.apply_move(bot_move)
   
        jboard_postion=Point(board_size+1-bot_move.point.row,bot_move.point.col) 
        bot_move_str = coords_from_point(jboard_postion)
        return jsonify({
            'bot_move': bot_move_str,
            'diagnostics': bot_agent.diagnostics()
        })

    return app


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
the_device = torch.device('cuda' if use_cuda else 'cpu')

number_of_planes = 10
board_size = 9
round_per_moves = 100

encoder = MultiplePlaneEncoder(number_of_planes, board_size)

input_shape = (number_of_planes, board_size, board_size)
model_new = Connect5Network(input_shape, board_size * board_size)
# model_new.load_state_dict(torch.load('./archived_model/new/1.pth'))
model_new.eval()


bot_agent = AlphaZeroAgent(2, "Agent_New", "2", encoder, model_new, round_per_moves, None, device=the_device)
human_agent = HumanPlayer(1, "HumanPlayerX", "1")

players = [bot_agent, human_agent]
start_player = human_agent

board = Board(board_size)
game = Connect5Game(board, players, start_player, False)


web_app = get_web_app({'alphazero': bot_agent})
web_app.run()
