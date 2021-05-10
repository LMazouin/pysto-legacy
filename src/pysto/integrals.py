#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy
from scipy.special import factorial as fact

from coefficients import *

EPS = 1.0E-6

def N(n, zeta):
    """
    normalization factor of the radial part  of a STO
    """
    return zeta**(n + 0.5) / numpy.sqrt(fact(2*n))


def A_func(x, n):
    """
    Auxiliary function A_n(x)
    """
    A = numpy.zeros(n+1, dtype=numpy.float64)
    C = numpy.zeros(n+1, dtype=numpy.float64)
    exp = numpy.exp(-x)
    C[0] = 1.0
    A[0] = exp / x
    for k in range(1, n+1):
        C[k] = x**k + k * C[k-1]
        A[k] = exp * C[k] / (x**(k+1))
    return A

def B_func(x, n):
    """
    Auxiliary function B_n(x)
    """
    sgn = numpy.sign(x)
    x = numpy.fabs(x)
    sphi = bessel(numpy.arange(n+1), x)
    B = numpy.zeros(n+1, dtype=numpy.float64)
    for k in range(0, n+1, 2):
        for j in range(0, k+1, 2):
            a = ANJ[k*(k+1)//2 + j]
            B[k] += 2.0 * a * sphi[j]
    for k in range(1, n+1, 2):
        for j in range(1, k+1, 2):
            a = ANJ[k*(k+1)//2 + j]
            B[k] += -sgn * 2.0 * a * sphi[j]
    return B

def Q_func(n, np, q, A, B):
    """
    Auxiliary function Q_nn'^q(p,pt)
    """
    Q = 0.0
    for m in range(n1+n2+1):
        Q += F[m,n,np] * A[n+np-m+q] * B[m+q]
    return Q

def rotation_matrix(lmax, e):
    """
    calculates the rotation matrix for atomic orbitals explicitely
    """

    T = numpy.zeros((2*lmax+1,2*lmax+1,lmax), dtype=numpy.float64)

    # calculate directional cosines and sinuses
    cost = e[2]
    sint = 0.0
    if (1.0 - cost**2) > EPS**2:
        sint = np.sqrt(1.0-cost**2)
    cosp = 1.0
    sinp = 0.0
    if sint > EPS:
        cosp = e[0]/sint
        sinp = e[1]/sint

    T[0,0,0] = 1.0

    # rotation of d orbitals
    #        dxy dyz dz2 dxz dx2-z2
    # dxy
    # dyz
    # dz2
    # dxz
    # dx2-y2

    if lmax == 2:

        SQRT3 = numpy.sqrt(3.0)

        sint2 = sint * sint
        cost2 = cost * cost
        sinp2 = sinp * sinp
        cosp2 = cosp * cosp

        cos2t = cost2 - sint2
        sin2t = 2.0 * sint * cost
        cos2p = cosp2 - sinp2
        sin2p = 2.0 * sinp * cosp

        T[lmax-2,lmax-2,2] =  cos2p * cost
        T[lmax-2,lmax-1,2] =  (1.0 - 2.0 * sinp2) * sint
        T[lmax-2,lmax  ,2] =  SQRT3 * sint2 * sinp * cosp
        T[lmax-2,lmax+1,2] =  2.0 * sint * cost * sinp * cosp
        T[lmax-2,lmax+2,2] =  (cost2 + 1.0) * sinp * cosp
        T[lmax-1,lmax-2,2] =  -sint * cosp
        T[lmax-1,lmax-1,2] =  cosp * cost
        T[lmax-1,lmax  ,2] =  SQRT3 * sinp * sin2t / 2.0
        T[lmax-1,lmax+1,2] =  sinp * cos2t
        T[lmax-1,lmax+2,2] =  -sinp * sin2t / 2.0
        T[lmax  ,lmax-2,2] =  0.0
        T[lmax  ,lmax-1,2] =  0.0
        T[lmax  ,lmax  ,2] =  1.0 - 3.0 * sint2 / 2.0
        T[lmax  ,lmax+1,2] =  -SQRT3 * sin2t / 2.0
        T[lmax  ,lmax+2,2] =  SQRT3 * sint2 / 2.0
        T[lmax+1,lmax-2,2] =  sinp * sint
        T[lmax+1,lmax-1,2] =  -sinp * cost
        T[lmax+1,lmax  ,2] =  SQRT3 * sin2t * cosp / 2.0
        T[lmax+1,lmax+1,2] =  cosp * cos2t
        T[lmax+1,lmax+2,2] =  -sin2t * cosp / 2.0
        T[lmax+2,lmax-2,2] =  -sin2p * cost
        T[lmax+2,lmax-1,2] =  -sint * sin2p
        T[lmax+2,lmax  ,2] =  SQRT3 * sint2 * cos2p / 2.0
        T[lmax+2,lmax+1,2] =  sin2t * cos2p / 2.0
        T[lmax+2,lmax+2,2] =  (1.0 - sint2 / 2.0) * cos2p

    # rotation of p orbitals
    #    py pz px
    # py
    # pz
    # px

    if lmax == 1:
        T[lmax-1,lmax-1,1] =  cosp
        T[lmax-1,lmax,  1] =  sinp * sint
        T[lmax-1,lmax+1,1] =  sinp * cost
        T[lmax,  lmax-1,1] =  0.0
        T[lmax,  lmax,  1] =  cost
        T[lmax,  lmax+1,1] =  -sint
        T[lmax+1,lmax-1,1] =  -sinp
        T[lmax+1,lmax,  1] =  sint * cosp
        T[lmax+1,lmax+1,1] =  cosp * cost

        return T

    if lmax == 0:
        return T
def rotate(lmax, lmin, la, lb, SS, T):
    """
    rotates the integrals from the aligned coordiante system to the molecular
    coordinate system
    """

    S = numpy.zeros((2*lmax+1,2*lmax+1), dtype=numpy.float64)

    if lmax == 0:
        S[0,0] = SS[0]
        return S

    ra = range(-la, la+1)
    rb = range(-lb, lb+1)
    s = 0.0
    for i, j in itertools.product(ra, rb):
        s = T[lmax+i,lmax,la] * SS[0] * T[lmax+j,lmax,lb]
        for k in range(1, lmin+1):
            s += SS[k] * ( T[lmax+i,lmax-k,la] * T[lmax+j,lmax-k,lb] + \
                           T[lmax+i,lmax+k,la] * T[lmax+j,lmax+k,lb] )
        S[la+i,lb+j] = s
    return S

def overlap(n, l, m, np, lp, mp, xyz, zet, zetp):
    """
    calculates the overlap between two STO
    """
    r = numpy.linalg.norm(xyz)
    # one-center overlap
    if r < EPS:
        if l == lp and m == mp:
            return fact(n + np) / (zet + zetp)**(n + np + 1)
        else:
            return 0.0
    # two-center overlap
    p = 0.5 * r * (zet + zetp)
    t = (zet - zetp) / (zet + zetp)
    # temporary variable for the phase factor (-1)^(l'+m')
    lptmp = lp
    # swap quantum numbers if necessary
    # since S(n,l,n',l',p,t) = S(n',l',n,l,p,-t)
    if lp > l or (lp == l and np > n):
        tmp = n
        n = np
        np = tmp
        tmp = l
        l = lp
        lp = tmp
        t = -t

    pt = p * t
    A = A_func(p,  n+np)
    B = B_func(pt, n+np)

    S = np.zeros(l1+1, dtype=np.float64)
    for lam in range(lp+1):
        phase = (-1.0)**(lptmp - lam)
        alfinf = -lam + (l  + lam) % 2
        betinf =  lam + (lp + lam) % 2
        ra = range(alfinf, l +1, 2)
        rb = range(betinf, lp+1, 2)
        for alf, bet in itertools.product(ra, rb):
            g0 = 0.0
            dbet = D[lp, lam, bet]
            kinf = max(0, (2*lam+alf-l)//2)
            ksup = min(lam, (alf+lam)//2) + 1
            for k in range(kinf, ksup):
                sgn = (-1.0)**k
                i = lam * (lam + 1) // 2 + k
                alk = alf + 2*lam - 2*k
                dalf = D[l1, lam, alk]
                g0 += sgn * B[i] * dalf
            g0 *= dbet
            al = alf + lam
            bl = bet - lam
            na = n1 - alf
            nb = n2 - bet
            for k in range(alf+bet+1):
                gq = g0 * F[k, al, bl]
                S[lam] += gq * Q_func(na, nb, k, A, B)
                #print(g0, gq, Q_func(na, nb, k, A, B))
        S[lam] *= phase * r**(n + np + 1)

    e = xyz / r
    lmin = min(l, lp)
    lmax = max(l, lp)
    T = rotation_matrix(lmax, e)
    S = rotate(lmax, lmin)
    return S[l+m, lp+mp]

