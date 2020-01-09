from agent.mctsagent import MCTSTree,MCTSNode
from agent.mctsagent import MCTSAgent
from agent.humanplayer import HumanPlayer
from common.board import Board
from common.utils import Utils
from game.connect5game import Connect5Game
import random


def evaluate():
    
    mcts_tree = MCTSTree()
    mcts_agent= MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "MCTSAgent", mcts_tree,1, 5.0)
    human_player = HumanPlayer(Connect5Game.ASSIGNED_PLAYER_ID_2,"Human")
        
    board = Board(8)
    
    players={}    
    players[mcts_agent.id]=mcts_agent
    players[human_player.id]= human_player

    game = Connect5Game(board,[mcts_agent.id, human_player.id],  random.choice(list(players.keys())),state_cache_size=0)
    
    
    while not game.is_over():
        move = players[game.working_game_state.player_in_action].select_move(game)
        if game.working_game_state.player_in_action== Connect5Game.ASSIGNED_PLAYER_ID_2:
            mcts_tree.go_down(move)
        
        game.apply_move(move)

        game.working_game_state.board.print_board()

    # game.working_game_state.board.print_board()
    # self._logger.info(game.final_winner.name)

    print(game.final_winner.name)


if __name__ == '__main__':
    evaluate()





