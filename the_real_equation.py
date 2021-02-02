import numpy as np
from numpy import arctan


def solution_theta_4(phi):
    k = 0.0825
    b = 0.316
    d = 0.384

    a2 = 4 * k**2 + (b-d)**2 - phi**2
    a1 = 4 * k * (b+d)
    a0 = (d+b)**2 - phi**2

    up = -a1 + np.sqrt(a1**2 - 4 * a0 * a2)
    down = 2 * a2
    theta = up/down
    print(theta)
    print(arctan(theta) * 2)
    return arctan(theta) * 2 

