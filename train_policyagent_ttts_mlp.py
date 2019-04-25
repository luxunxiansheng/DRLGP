
import h5py
from tqdm import tqdm

from agent.policyagent import PolicyAgent
from common.experiencebuffer import ExpericenceBuffer
from common.experiencecollector import ExperienceCollector
from common.oneplaneencoder import OnePlaneEncoder
from game.tictactoe.tictactoegame import TicTacToeGame


def main():

    board_size = 3
    num_game = 500

    encoder = OnePlaneEncoder(board_size)
    collector1 = ExperienceCollector()
    pgAgent1 = PolicyAgent(0,'PG1',encoder, collector1)

    collector2 = ExperienceCollector()
    pgAgent2 = PolicyAgent(1, 'PG2', encoder, collector2)

    players = [pgAgent1, pgAgent2]

    start_player = players[1]

    for _ in tqdm(range(num_game)):
        collector1.begin_episode()
        collector2.begin_episode()

        game_winner = TicTacToeGame.simulate(board_size, players, start_player)
        if game_winner == pgAgent1:
            collector1.compelte_episode(reward=1)
            collector2.compelte_episode(reward=-1)
        else:
            collector1.compelte_episode(reward=-1)
            collector2.compelte_episode(reward=1)

    experience = ExpericenceBuffer.combine_experience([collector1, collector2])

    
    experience.serialize('./experience/pg.tar')


if __name__ == "__main__":
    main()
