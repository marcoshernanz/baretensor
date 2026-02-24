# %%

import bt
import torch
import numpy as np

ndarray = np.arange(10, dtype=np.float32)

print(ndarray)


t1 = torch.tensor(ndarray)
print(t1)

t2 = bt.tensor(ndarray)
print(t2)
