import ctypes as c
import os as os

import numpy as np

___ctypes_local_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
___lib_c = c.CDLL(___ctypes_local_dir + "libfgn.so")
___lib_c.apprcirc.argtypes = [
    c.c_long,
    c.c_double,
    c.c_double,
    c.c_int,
    c.POINTER(c.c_double),
]
___lib_c.apprcirc.restype = None


def generate_fgn(n_points: int, *, hurst: float = 0.5, seed: int = -1) -> np.ndarray:
    """Generate fractional Gaussian noise.

    Interface to C implementation of the approximate circulant method.

    Input:
        n_points:
            Number of points to generate (will be increased to match
                closest power of 2).
        hurst: (default: 0.5)
            Desired Hurst exponent.
        seed:
            Seed for the RNG (default: -1). Negative values will prompt
            the seed to be auto-generated.

    Output:
        n_points samples of the fractional Gaussian noise. Samples
        should have zero mean and unit standard deviation (they are
        generated so that this would be true while averaging these
        statistical properties over multiple runs).
    """
    # auto-generate seed
    if seed < 0:
        np.random.seed()
        seed = np.random.randint(0, int(2 ** 20))

    # recalculate n_points so that the value would be a power of 2
    log2_n = np.floor(np.log2(n_points)).astype(int)
    fgn_points = 2 ** log2_n
    while fgn_points < n_points:
        fgn_points = fgn_points * 2
        log2_n = log2_n + 1

    # generate fractional Gaussian noise
    data = (c.c_double * fgn_points)()
    ___lib_c.apprcirc(log2_n, hurst, c.c_double(fgn_points), seed, data)
    ret = np.array([d for d in data])
    del data

    # pick subset of generated points if desired number of points is smaller
    if n_points < fgn_points:
        ret = ret[:n_points]

    return ret
