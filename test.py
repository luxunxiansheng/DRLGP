class A(object):
    def __init__(self):
        self._name = "hello"

    def show(self):
        print(self._name)
        
    def change_name(self):
        self._name="world"    

a1 = A()
a1.show()

b1 = a1
b1.change_name()
b1.show()


a1.show()

c1 = a1
c1.show()




