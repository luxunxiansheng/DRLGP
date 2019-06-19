from common.point import Point


class Piece:
    def __init__(self,owner_id, point):
        self._owner_id = owner_id
        self._point = point

    @property
    def owner_id(self):
        return self._owner_id

    @property
    def point(self):
        return self._point
