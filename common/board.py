from common.point import Point

class Board:
    
    alphabet = list(map(chr,range(65,91)))    
      
    def __init__(self,board_size):
        self._board_size = board_size
        self._column_indicator=['  %s'%Board.alphabet[i] for i in range(0,board_size)]
        self._rows=tuple(range(1, board_size+1)) 
        self._cols=tuple(range(1, board_size+1)) 
        self._grid = {}

       
    def is_on_grid(self,point):
        return 1 <= point.row <= self._board_size and 1 <= point.col <= self._board_size

    def get_player(self, point):
        return self._grid.get(point)

    def place(self, player, point):
        assert self.is_on_grid(point)
        assert self.get_player(point) is None

        self._grid[point] = player


    def print_board(self):
        print('**************************************************')
              
        print(self._column_indicator)

        for row in range(1,self._board_size+1):
            pieces = []
            for col in range(1,self._board_size+1):
                player = self.get_player(Point(row, col))
                pieces.append(player.mark) if player is not None else pieces.append('')
           
        
        print('%d %s' % (row, ' | '.join(pieces)))


class Move:
    def __init__(self, point):
        self.point = point


class GameState:
    def __init__(self, board, the_player, move):
        self.board = board
        self.the_player = the_player
        self.last_move = move

    def apply_move(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.place(self.the_player, move.point)
        return GameState(next_board, self.the_player.other, move)

    @classmethod
    def new_game(cls,board_size):
        board = Board(board_size)
        return GameState(board, Player.x, None)

    def is_valid_move(self, move):
        return(self.board.get(move.point) is None and not self.is_over())

    def legal_moves(self):
        moves = []
        for row in self.board._rows:
            for col in self.board._cols:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves

    def is_over(self):

        if self._in_a_row(Player.x):
            return True

        if self._in_a_row(Player.o):
            return True

        if all(self.board.get(Point(row, col)) is not None
               for row in self.board._rows
               for col in self.board._cols):
            return True

        return False

    def _in_a_row(self, player):
        for col in self.board._cols:
            if all(self.board.get(Point(row, col)) == player for row in self.board._rows):
                return True

        for row in self.board._rows:
            if all(self.board.get(Point(row, col)) == player for col in self.board._cols):
                return True

        # Diagonal RL to LR
        if all(self.board.get(Point(i,i)) == player for i in range(1,self.board._board_size+1)):
            return True

        if all(self.board.get(Point(i,self.board._board_size+1-i)) == player for i in range(1,self.board._board_size+1)):
            return True
     

        return False
