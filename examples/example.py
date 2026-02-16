# %%
import sys
import os

sys.path.append(os.path.abspath("../src"))

import bt

t = bt.Tensor([10, 20])
print(t.shape)
