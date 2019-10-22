
import numpy as np

for _ in range(100):
    choice = np.array([[0, 0, 0], [2, 2, 2],[3,3,3],[4,4,4]])

    idx = np.random.choice(len(choice), p=[0.1,0.1,0.1,0.7])

    print(idx)








