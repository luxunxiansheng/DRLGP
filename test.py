from agent.mctsagent import MCTSTree,MCTSNode
from agent.mctsagent import MCTSAgent
from agent.humanplayer import HumanPlayer
from common.board import Board
from common.utils import Utils
from game.connect5game import Connect5Game
import random


def evaluate():
    
    mcts_tree = MCTSTree()
    mcts_agent= MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "MCTSAgent", mcts_tree,100, 1.4)

    human_player = HumanPlayer(Connect5Game.ASSIGNED_PLAYER_ID_2,"Human")
        
    board = Board(8)
    players = [mcts_agent, human_player]
    game = Connect5Game(board, players,  players[random.choice([0, 1])])


    while not game.is_over():
        move = game.working_game_state.player_in_action.select_move(game)
        if game.working_game_state.player_in_action.id == Connect5Game.ASSIGNED_PLAYER_ID_2:
            mcts_tree.go_down(move)
        
        game.apply_move(move)

        game.working_game_state.board.print_board()

    # game.working_game_state.board.print_board()
    # self._logger.info(game.final_winner.name)

    print(game.final_winner.name)


if __name__ == '__main__':
    evaluate()





