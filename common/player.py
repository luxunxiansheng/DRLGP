class Player:
    def __init__(self, id, name, mark):
        self._id = id
        self._name = name
        self._mark = mark

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def mark(self):
        return self._mark
