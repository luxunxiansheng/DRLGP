class Player(object):
    def __init__(self, id, name,mark):
        self._id = id
        self._name = name
        self._mark = mark
        self._diagnostics_info =''
           
    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name
    
    @property
    def mark(self):
        return self._mark


    def select_move(self,game,game_state):
        raise NotImplementedError()

    def __eq__(self,other):
        if not isinstance(other,Player):
            return NotImplemented
        
        return self.id == other.id
    @property    
    def diagnostics(self):
        return self._diagnostics_info

    @diagnostics.setter
    def diagnostics(self, info):
        self._diagnostics_info = info      