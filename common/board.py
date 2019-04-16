import copy

from common.point import Point


class Board:

    alphabet = list(map(chr, range(65, 91)))
    
    def __init__(self, board_size):
        self._board_size = board_size
        self._column_indicator = ['  %s' % Board.alphabet[i]
                                  for i in range(0, board_size)]
        self._rows = tuple(range(1, board_size+1))
        self._cols = tuple(range(1, board_size+1))
        self._grid = {}

    @classmethod
    def get_column_indicator_index(cls,char):
        return Board.alphabet.index(char) 

    def is_on_grid(self, point):
        return 1 <= point.row <= self._board_size and 1 <= point.col <= self._board_size

    def get_player(self, point):
        return self._grid.get(point)

    def place(self, player, point):
        assert self.is_on_grid(point)
        assert self.get_player(point) is None

        self._grid[point] = player

        
    
    def print_board(self):
        print('**************************************************')

        print(''.join(self._column_indicator))

        for row in range(1, self._board_size+1):
            pieces = []
            for col in range(1, self._board_size+1):
                player = self.get_player(Point(row, col))
                pieces.append(player.mark) if player is not None else pieces.append('')
            print('%d %s' % (row, ' | '.join(pieces))) 
    
