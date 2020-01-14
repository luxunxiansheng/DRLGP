import random

from agent.humanplayer import HumanPlayer
from agent.mctsagent import MCTSAgent
from common.board import Board
from common.gamestate import GameState
from common.utils import Utils
from game.connect5game import Connect5Game
from memory_profiler import profile


def evaluate():
    
    mcts_agent= MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "MCTSAgent",2500, 5.0)
    human_player = HumanPlayer(Connect5Game.ASSIGNED_PLAYER_ID_2,"Human")
        
    board = Board(8)
    
    players={}    
    players[mcts_agent.id]=mcts_agent
    players[human_player.id]= human_player

    init_game_state= GameState(board,0,None)
    game = Connect5Game(init_game_state,[mcts_agent.id,human_player.id],0)

    while not game.is_over():
        move = players[game.working_game_state.player_in_action].select_move(game)
        if game.working_game_state.player_in_action== Connect5Game.ASSIGNED_PLAYER_ID_2:
            mcts_agent.mcts_tree.go_down(game,move)
        
        game.apply_move(move)
        print("Last move is {}".format(move.point))

        if game.working_game_state.player_in_action== Connect5Game.ASSIGNED_PLAYER_ID_1: 
            game.working_game_state.board.print_board()

    game.working_game_state.board.print_board()
    # self._logger.info(game.final_winner.name)

    print(players[game.final_winner].name)


if __name__ == '__main__':
    evaluate()
