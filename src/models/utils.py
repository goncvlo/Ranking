import numpy as np
import random


def set_global_seed(seed: int = 42):
    #os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
