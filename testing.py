from random import uniform
from time import time

import numpy as np

T = 1234567

def numpy_1():
    n2 = np.random.rand(T)
    for x in range(0, T):
        np.random.choice(n2)

    return []


def numpy_2():
    n2 = np.random.rand(T)
    return n2


def umiform_1():
    for x in range(0, T):
        uniform(0, 1)

    return []


if __name__ == '__main__':
    t1 = time()
    a = numpy_1()

    t2 = time()
    print t2 - t1
    b = numpy_2()
    t3 = time()
    print t3 - t2

    c = umiform_1()
    t4 = time()
    print t4 - t3
