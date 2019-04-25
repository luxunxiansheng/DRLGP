class Player(object):
    def __init__(self, id, name):
        self._id = id
        self._name = name
           
    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name


    def select_move(self,game,game_state):
        raise NotImplementedError()

    def __eq__(self,other):
        if not isinstance(other,Player):
            return NotImplemented
        
        return self.id == other.id 