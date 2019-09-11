
import numpy as np

choice = np.array([[0, 0, 0], [2, 2, 2],[3,3,3],[4,4,4]])

idx = np.random.choice(len(choice), p=[0.1,0.2,0.3,0.4])

print(choice[idx])






