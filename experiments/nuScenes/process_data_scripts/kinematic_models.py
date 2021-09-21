import numpy as np
import sys
sys.path.append("../../../trajectron")
from environment import derivative_of_new as derivative_of


def calc_kinematic(seq, data_mode='last', dt=0.5, alpha=None, radian=False):
    rate_seq = derivative_of(seq, dt, radian=radian)

    if data_mode == 'last':
        rate = rate_seq[-1]

    elif data_mode == 'mean':
        rate = rate_seq.mean()

    elif data_mode == 'smooth':
        alpha = 2.0 / (len(rate_seq) + 1)
        rate = rate_seq[0]
        for t in range(1, len(rate_seq)):
            rate = alpha * rate_seq[t] + (1 - alpha) * rate
    elif data_mode == 'smooth_alpha_0_6':
        alpha = 0.6
        rate = rate_seq[0]
        for t in range(1, len(rate_seq)):
            rate = alpha * rate_seq[t] + (1 - alpha) * rate
    else:
        raise NotImplementedError

    return rate
