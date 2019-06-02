from collections import namedtuple


class Point(namedtuple('Point', 'row col')):
    def __deepcopy__(self, memodict={}):
        return self
    def __eq__(self,other):
        return self.row == other.row and self.col == other.col