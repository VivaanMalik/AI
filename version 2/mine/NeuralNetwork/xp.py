import os
import sys

os.environ["USE_CUPY"] = "true"

def setup():
    import numpy as xp

    sys.modules["xp"] = xp