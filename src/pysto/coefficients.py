#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy
from scipy.special import binom
from scipy.special import factorial2 as fact
from scipy.special import factorial2 as fact2
from scipy.special import spherical_in as bessel

NMAX = 6
LMAX = 3

# generate and store a list with binomial coefficients
B = numpy.zeros((NMAX+1)*(NMAX+2)//2, dtype=numpy.float64)
for n in range(NMAX+1):
    for k in range(n+1):
        i = n * (n+1) // 2 + k
        B[i] = binom(n, k)


def F_func(m, n, np):
    """
    calculates generalized binomial coefficients
    """
    kinf = ((m - n) + abs(m - n))//2
    ksup = min(m, np) + 1
    f = 0.0
    for k in range(kinf, ksup):
        i = n * (n + 1) // 2 + m - k
        j = np * (np + 1) // 2 + k
        f += (-1.0)**k * B[i] * B[j]
    return f

def D_func(l, lam, bet):
    """
    calculates quantities needed by the g0 coefficients (see below)
    """
    lbet = (l - bet) // 2
    i = (l + lam) * (l + lam + 1) // 2 + l
    j = l * (l + 1) // 2 + lam
    m = l * (l + 1) // 2 + lbet
    n = (l + bet) * (l + bet + 1) // 2 + bet - lam
    term1  = 0.5**l
    term2  = (-1.0)**lbet
    term3 = numpy.sqrt((2*l + 1) / 2) * numpy.sqrt(B[i] / B[j])
    term4 = B[m] * B[n]
    return term1 * term2 * term3 * term4

def G0_func(l, lp, lam, alf, bet):
    """
    calculates expansion coefficients of normalized associeted Legendre in
    elliptical coordinates
    """
    s = 0.0
    dbet = D_function(lp, lam, bet)
    kinf = max(0, (2*lam + alf - l)//2)
    ksup = min(lam, (alf + lam)//2) + 1
    for k in range(kinf, ksup):
        sgn = (-1.0)**k
        i = lam * (lam + 1) // 2 + k
        alk = alf + 2*lam - 2*k
        dalf = D_function(l, lam, alk)
        s += sgn * B[i] * dalf
    return s * dbet

F = numpy.zeros((2*NMAX+1, NMAX+1, NMAX+1), dtype=numpy.float64)
r = range(NMAX+1, NMAX+1)
for n, np in itertools.product(r, r):
    for m in range(n+np+1):
        F[m,n,np] = F_func(m, n, np)

D = numpy.zeros((LMAX+1, LMAX+1, LMAX+1), dtype=numpy.float64)
for l in range(LMAX+1):
    for lam in range(l+1):
        betinf = lam + (l + lam)%2
        for bet in range(betinf, l+1, 2):
            D[l,lam,bet] = D_func(l, lam, bet)

ANJ = numpy.zeros((NMAX+1)*(NMAX+2)//2, dtype=numpy.float64)
for n in range(NMAX):
    for j in range(n+1):
        ind = n * (n + 1) // 2 + j
        ANJ[ind] = fact(n) * (2.0*j + 1.0) / (fact2(n-j) * fact2(n+j+1))

if __name__ == '__main__':
    pass
