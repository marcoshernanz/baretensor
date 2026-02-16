# %%
import sys
import os

sys.path.append(os.path.abspath("../src"))

import bt
import torch

# %%

t1 = torch.ones(1, 2, 3, 4)
print(t1.shape)

t2 = bt.ones([10, 20, 30])
print(t2.shape)
