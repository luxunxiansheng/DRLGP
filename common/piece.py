from common.point import Point


class Piece:
    def __init__(self,owner, point):
        self._owner = owner
        self._point = point

    @property
    def owner(self):
        return self._owner

    @property
    def point(self):
        return self._point
