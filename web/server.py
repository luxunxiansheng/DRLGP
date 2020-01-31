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
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from models.ResNet8Network import ResNet8Network
from game.connect5game import Connect5Game
from common.point import Point
from common.move import Move
from common.gamestate import GameState
from common.board import Board
from boardencoder.deepmindencoder import DeepMindEncoder
from agent.humanplayer import HumanPlayer
from agent.alphazeroagent import AlphaZeroAgent
from flask import Flask, jsonify, request
import torch



def coords_from_point(point):
    return '%s%d' % (Board.get_column_indicator(point.col-1), point.row)


def point_from_coords(text):
    col_name = text[0]
    row = int(text[1])
    return Point(row, Board.get_column_indicator_index(col_name)+1)


def get_web_app():
    """Create a flask application for serving bot moves.
    Returns: Flask application instance
    """
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, 'static')

    app = Flask(__name__, static_folder=static_path, static_url_path='/static')

    @app.route('/select-move/<bot_name>', methods=['POST'])
    def select_move(bot_name):

        bot_agent = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "Agent_New", encoder, model_new, az_mcts_round_per_moves, c_puct, az_mcts_temperature, device=device)
        human_agent = HumanPlayer(Connect5Game.ASSIGNED_PLAYER_ID_1, "HumanPlayerX")

        board = Board(board_size)
        players = {}
        players[bot_agent.id] = bot_agent
        players[human_agent.id] = human_agent

        start_game_state = GameState(board, human_agent.id, None)

        game = Connect5Game(start_game_state, [bot_agent.id, human_agent.id], number_of_planes, False)

        historic_jboard_positions = [point_from_coords([move][0]) for move in (request.json)['moves']]
        historic_moves = [Move(Point(board_size+1-historic_jboard_position.row, historic_jboard_position.col)) for historic_jboard_position in historic_jboard_positions]
        over = False
        bot_move_str = ''

        for move in historic_moves:
            game.apply_move(move)

        if game.is_over():
            over = True
        else:
            bot_move = bot_agent.select_move(game)
            game.apply_move(bot_move)
            game.working_game_state.board.print_board()
            jboard_postion = Point(board_size+1-bot_move.point.row, bot_move.point.col)
            bot_move_str = coords_from_point(jboard_postion)
            over = True if game.is_over() else False

        return jsonify({
            'bot_move': bot_move_str,
            'over': over,
            'diagnostics': None
        })

    return app


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

number_of_planes = 4
board_size = 8
az_mcts_round_per_moves = 700
c_puct = 8.0
az_mcts_temperature = 0.001

encoder = DeepMindEncoder(number_of_planes, board_size)

input_shape = (number_of_planes*2+1, board_size, board_size)
model_new = ResNet8Network(input_shape, board_size * board_size)


best_checkpoint_file = './checkpoints/test/latest.pth.tar'
checkpoint = torch.load(best_checkpoint_file, map_location='cpu')
model_new.load_state_dict(checkpoint['model'])
model_new.eval()


web_app = get_web_app()
web_app.run()
