class Move:
    def __init__(self, player,point):
        self._point = point
        self._player= player

    @property
    def point(self):
        return self._point
    
    @property
    def player(self):
        return self._player
