import multiprocessing as mp

class MyFancyClass(object):
    
    def __init__(self,i):
        self.names = []
        self.i=i
    
    def do_something(self):
        for i in range(5000000):
            self.names.append(str(self.i))
            
        print('----------------------------------------------------------')


def worker(pipe,i):
    myclass = MyFancyClass(i)
    myclass.do_something()
    pipe.send(myclass)
    pipe.close()



if __name__ == '__main__':
    scores=[]
    mp.set_start_method('spawn')
    
    
    processes = []
    pipes = []
    for i in range(3):
        parent_conn,child_conn = mp.Pipe()
        p = mp.Process(target=worker, args=(child_conn,i))
        processes.append(p)
        pipes.append(parent_conn)
        p.start()
        
     
    
    scores = [parent_conn.recv() for parent_conn in pipes]
        
    for p in processes:
        p.join()

     

    for pipe in pipes:
        pipe.close()
       
        print('*****************************************************************')
    
      

    for score in scores:
        print(score.names)

    