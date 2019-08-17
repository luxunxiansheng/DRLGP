import multiprocessing as mp

class MyFancyClass(object):
    
    def __init__(self,i):
        self.names = []
        self.i=i
    
    def do_something(self):
        for i in range(5000000):
            self.names.append(str(self.i))
            
        print('----------------------------------------------------------')


def worker(results,i):
    myclass = MyFancyClass(i)
    myclass.do_something()
    results.put(myclass)
    

if __name__ == '__main__':
    scores=[]
    mp.set_start_method('spawn')
    
    
    processes = []
    results = mp.Queue()

    for i in range(3):
        
        p = mp.Process(target=worker, args=(results,i))
        processes.append(p)
        p.start()

        scores.append(results.get())   
     
    
    
    for p in processes:
        p.join()

       
    print('*****************************************************************')
    
      

    for score in scores:
        print(score.names)

    