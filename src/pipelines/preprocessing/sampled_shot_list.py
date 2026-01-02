import numpy as np 
import random


# ----------------------------------------------------------------------------------------------------------------------
def yamane_sampled_shot_list(shot_list, error=0.05):
    N = len(shot_list)
    n = int(np.round(N/(1+N*(error**2))))
    return random.sample(shot_list, n)
    