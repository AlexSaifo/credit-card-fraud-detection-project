import numpy as np

def calc_delta(pre_derivative_sign):
    return (pre_derivative_sign * pre_derivative_sign * -0.15) + (0.35 * pre_derivative_sign) + 1
