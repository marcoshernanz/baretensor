# %%
import sys
import os

sys.path.append(os.path.abspath("../src"))

import bt

t = bt.ones([10, 20, 30])
print(t.shape)
