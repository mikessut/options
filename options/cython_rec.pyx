"""
Cython recursive function
"""

import numpy as np
cimport numpy as np
np.import_array()
import cython
from libc.math cimport sqrt, log, exp, abs

def test():
    a: cython.int = 0
    i: cython.int
    for i in range(10):
        a += i
    print(a)


def garch_create_lrs(np.ndarray[double, ndim=2] lrs, 
                     double var0, double lr0, int num_sims, int max_steps, 
                     double w, double alpha, double beta):
    n: cython.int
    nd: cython.int
    var: double
    sr: double
    cdef np.ndarray[double, ndim=1] r
    for n in range(num_sims):
        var = var0
        sr = lr0
        r = np.random.randn(max_steps)
        for nd in range(max_steps):
            var = w + alpha * sr*sr + beta * var
            with cython.boundscheck(False):
                sr = sqrt(var) * r[nd]
            lr = log(1 + sr)
            with cython.boundscheck(False):
                lrs[nd, n] = lr


def egarch_create_lrs(np.ndarray[double, ndim=2] lrs, 
                     double var0, double lr0, int num_sims, int max_steps, 
                     double omega, double alpha, double theta, double lam):
    n: cython.int
    nd: cython.int
    var: double
    sr: double
    lr: double
    sig2: double
    lsig2: double
    sig: double
    e: double
    cdef np.ndarray[double, ndim=1] r
    for n in range(num_sims):
        var = var0
        lr = lr0
        sig2 = var0
        lsig2 = log(sig2)
        sig = sqrt(exp(lsig2))
        r = np.random.randn(max_steps)
        
        for nd in range(max_steps):
            e = lr / sig
            lsig2 = omega + theta * e + lam * (abs(e) - sqrt(2 / 3.141592653589793)) + alpha * lsig2
            sig = sqrt(exp(lsig2))
            
            with cython.boundscheck(False):
                sr = sig * r[nd]
            lr = log(1 + sr)
            with cython.boundscheck(False):
                lrs[nd, n] = lr