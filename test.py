import multiprocessing as mp
import random
import numpy as np
from time import time
from multiprocessing import Process, Queue


print("Number of processors: ", mp.cpu_count())

def howmany_within_range(queue,x):
    r = x + random.choice([1, 2, 3])
    print(r)
    queue.put(r)


queue= Queue()

processes = [Process(target=howmany_within_range, args=(queue,100)) for _ in range(4)]

for p in processes:
    p.start()


for p in processes:
    p.join()

results = [queue.get() for _ in processes]

print(results)