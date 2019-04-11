from collections import namedtuple

class Point(namedtuple('Point', 'row col')):
    def __deepcopy__(self, memodict={}):
        return self