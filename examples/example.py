# %%
import sys
import os

sys.path.append(os.path.abspath("../src"))

import bt
import torch


t1 = torch.ones([2, 3, 4])
print(t1)

t2 = bt.ones([2, 3, 4])
print(t2)
