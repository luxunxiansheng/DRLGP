from .point import Point


class Move:
    def __init__(self, point):
        self._point = point

    @property
    def point(self):
        return self._point
