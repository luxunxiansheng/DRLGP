# DRLGP  
Deep Reinforcement Learning Game Playground  
<br>

# Table of Contents
* [Introduction](#Introduction) 
* [Basic Algorithms](#BasicAlgorithms)
* [Implementation notes](#Implementation notes)
* [Usage](#Usage)
* [References](#References)
* [License](#License)

## Introduction

Board games can be classified according to useful properties:

* Deterministic vs. nondeterministic 
  In a deterministic game, the course of the game depends only on the players' decisions. In a nondeterministic game, an element of randomness is involved,
  such as shuffling cards. Go game is deterministic,typically

* Perfect information vs. hidden information 
  In *perfect information games*, both players can see the full game state at 
  all times; the whole board is visible. *In hidden information games*,each 
  player can see only part of the game states. JunQi, a popular Chinese board 
  game, is typical for opponent'pieces are kept unkonwn in the game process. 


DRLGP is a platform to help developing board game agents strengthened with 
tree search methods. As one of examples, AlphaGo,which defeated human top Go 
player in 2016,is a reinforcement learning and tree search based bot. 

On DRLGP, deterministic and perfect information board game bots will be 
developed first, and then hidden information game bot.


## Basic Algorithms
1. [Minimax](https://en.wikipedia.org/wiki/Minimax)
<br>
Minimax is a decision rule used in artificial intelligence, decision theory, game theory, statistics and philosophy for minimizing the possible loss for a worst case (maximum loss) scenario. When dealing with gains, it is referred to as "maximin"—to maximize the minimum gain. Originally formulated for two-player zero-sum game theory,covering both the cases where players take alternate moves and those where they make simultaneous moves, it has also been extended to more complex games and to general decision-making in the presence of uncertainty.


2. [Alpha-Beta pruning](https://en.wikipedia.org/wiki/Alpha-beta_pruning)
<br>
Alpha–beta pruning is a search algorithm that seeks to decrease the number of nodes that are checkd by the minimax algorithm in its search tree. It is an adversarial search algorithm used commonly for machine playing of two-player games (Tic-tac-toe, Chess, Go, etc.). It stops evaluating a move when at least one possibility has been found that proves the move to be worse than a previously examined move. Such moves need not be checkd further. When applied to a standard minimax tree, it returns the same move as minimax would, but prunes away branches that cannot possibly influence the final decision

3. [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) 
<br>
In computer science, Monte Carlo tree search (MCTS) is a heuristic search algorithm for some kinds of decision processes, most notably those employed in game play. MCTS has been used for decades in computer Go programs.It has been used in other board games like chess and shogi,games with incomplete information such as bridge and poker,as well as in real-time video games (such as Total War: Rome II's implementation in the high level campaign AI.


4. UCT 

5. ResNet


## Implementation notes 

1.  pytorch 1.0+


## Usage
  1.  To run the train pipeline, modify the config.ini to setup your favoriate parameters
      and then launch the entry point in the root folder:
    
       ```
        python   main.py
       
       ```
  2.  To play the game, run the following command to launch the web server 
       ```
        python   ./web/server.py 
       
       ```

      and then input the 127.0.0.1:5000/static/index.html into the web browser to open the game page.

## References
[1] [Silver et al. “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.”](https://arxiv.org/pdf/1712.01815.pdf)

[2] [Silver et al. “Mastering the game of go without human knowledge.” Nature 550.7676 (2017): 354–359.](https://www.gwern.net/docs/rl/2017-silver.pdf)

[3] [Max Pumperla and Kevin Ferguson."Deep Learning and the Game of Go"](https://www.manning.com/books/deep-learning-and-the-game-of-go)

[4] [Maxim Lapan."Deep Reinforcement Learning Hands-On"](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands)

[5] [宋俊潇,AlphaZero实战：从零学下五子棋](https://zhuanlan.zhihu.com/p/32089487)




## License
[MIT](https://choosealicense.com/licenses/mit/)


