class Encoder:
    def name(self):
        raise NotImplementedError()
    
    def encode(self,game_state):
        raise NotImplementedError()

    def decode(self,board_matrix):
        raise NotImplementedError    
    
    def encode_point(self,point):
        raise NotImplementedError()

    def decode_point_index(self,index):
        raise NotImplementedError()

    def num_points(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()       
    
    def board_width(self):
        raise NotImplementedError()       
    
    def board_height(self):
        raise NotImplementedError()       
    

