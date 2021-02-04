#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools as it
import numpy as np

from scipy.special import factorial as fact
from scipy.special import binom
from scipy.integrate import quad

NMAX = 8

F = np.zeros((NMAX+1)*(NMAX+2)//2, dtype=np.float64)
for n in range(NMAX):
    for k in range(n+1):
        ind = n * (n+1) // 2 + k
        F[ind] = binom(n, k)
EPS = 1.0E-15

def rotation_matrix(lmax, e):
    """
    Explicit calculation of the rotation matrix of atomic orbitals.
    """

    T = np.zeros((2*lmax+1,2*lmax+1,lmax), dtype=np.float64)

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

        SQRT3 = np.sqrt(3.0)

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
    Rotation of the integrals from the aligned to the molecular frame.
    """

    S = np.zeros((2*lmax+1,2*lmax+1), dtype=np.float64)

    if lmax == 0:
        S[0,0] = SS[0]
        return S

    ra = range(-la, la+1)
    rb = range(-lb, lb+1)
    s = 0.0
    for i, j in it.product(ra, rb):
        s = T[lmax+i,lmax,la] * SS[0] * T[lmax+j,lmax,lb]
        for k in range(1, lmin+1):
            s += SS[k] * ( T[lmax+i,lmax-k,la] * T[lmax+j,lmax-k,lb] + \
                           T[lmax+i,lmax+k,la] * T[lmax+j,lmax+k,lb] )
        S[la+i,lb+j] = s
    return S

def normalize(na, nb, p, t):
    """
    Normalization factor of the integrals.
    """

    num = p**(na+nb+1) * (1.0+t)**na * (1.0-t)**nb * np.sqrt(1.0-t**2)
    den = np.sqrt(fact(2*na) * fact(2*nb))

    return num / den

def A_func(n, p):
    """
    Auxiliary function A_n(p).
    """

    A = np.zeros(n+1, dtype=np.float64)
    C = np.zeros(n+1, dtype=np.float64)
    exp = np.exp(-p)
    C[0] = 1.0
    A[0] = exp / p
    for k in range(1, n+1):
        C[k] = p**k + k * C[k-1]
        A[k] = exp * C[k] / (p**(k+1))
    return A

def B_func(n, pt):
    """
    Auxiliary function B_n(pt).
    """

    B = np.zeros(n+1, dtype=np.float64)
    exp = np.exp(pt)
    iexp = 1.0 / exp

    # use the recursive algorithm to compute B for large pt
    if abs(pt) > 3.0:
        B[0] = (exp - iexp) / pt
        for k in range(1, n+1):
            B[k] += (k * B[k-1] + (-1.0)**k * exp - iexp) / pt
        return B

    # use a Taylor series to calculate B for small pt 
    IMAX = 30
    if abs(pt) > EPS:
        for k in range(0, n+1):
            #s0 = 0.0
            s1 = 0.0
            for i in range(0, IMAX):
                s1 += (-pt)**(i) * (1.0 + (-1.0)**(i+k)) / (fact(i)*(i+k+1.0))
                #print(i, s0, s1)
                #if abs(s0 - s1) < EPS:
                #    break
                #s0 = s1
            B[k] = s1
        return B

    if abs(pt) < EPS:
        for k in range(0, n+1):
            B[k] = (1.0 + (-1.0)**k) / (k+1.0)
        return B

def Q_function(na, nb, k, A, B):
    """
    Auxiliary function Q^q_NN'(p,t).
    """

    Q = 0.0
    for m in range(na+nb+1):
        iinf = ((m - na) + abs(m - na))//2
        isup = min(m, nb) + 1
        FF = 0.0
        for i in range(iinf, isup):
            ind1 = na*(na + 1)//2 + m - i
            ind2 = nb*(nb + 1)//2 + i
            FF += (-1.0)**i * F[ind1] * F[ind2]
        # more Pythonic approach
        # i = np.arange(iinf, isup)
        # ind1 = na*(na + 1)//2 + m - i
        # ind2 = nb*(nb + 1)//2 + i
        # f = (-1.0)**i * F[ind1] * F[ind2]
        # FF = f.sum() 
        k1 = na + nb - m + k
        k2 = m + k
        Q += FF * A[k1] * B[k2]
        #print(k1, k2, A[k1], B[k2])
    return Q

def test():
    """
    Test to assert the precision of the algorithm for the B function.
    Comparison with Gaussian quadrature
    """
    f = lambda x, k, a : x**k * np.exp(-a*x)
    h = 0.1
    for n in range(0, 6):
        print('n = ', n)
        for i in range(0, 101):
            pt = h * i
            B1 = B_func(n, pt)
            B2, err = quad(f, -1.0, 1.0, args=(n,pt),  limit=500, epsabs=EPS, epsrel=EPS, maxp1=500, limlst=500)
            error = abs(B1[n] - B2)
            print('{0:6.2f} {1:20.15f} {2:20.15f} {3:10.3E}'.format(pt, B1[n], B2, error))

def basic_overlap(na, la, nb, lb, p, t):

    # temporary variable for the phase factor (-1)^(lb+mb)
    lbtemp = lb

    # swap quantum numbers if necessary
    # since S(nl,n'l',p,t) = S(n'l',nl,p,-t)
    if lb > la or (lb == la and nb > na):
        temp = na
        na = nb
        nb = temp
        temp = la
        la = lb
        lb = temp
        t = -t

    n = na + nb
    pt = p * t
    A = A_func(n, p)
    B = B_func(n, pt)

    SS = np.zeros(la+1, dtype=np.float64)
    # s - s overlap
    if la == 0:
        SS[0] += 0.5 * Q_function(na, nb, 0, A, B)
        return SS

    # s - p overlap
    if la == 1 and lb == 0:
        phase = (-1.0)**lbtemp
        factor = 0.5*np.sqrt(3.0)
        SS[0] += factor * Q_function(na-1, nb, 0, A, B)
        SS[0] += factor * Q_function(na-1, nb, 1, A, B)
        SS[0] *= phase
        return SS

    # p - p overlap
    if la == 1 and lb == 1:
        # p - p sigma
        phase = (-1.0)**lbtemp
        factor = 1.5
        SS[0] +=  factor * Q_function(na-1, nb-1, 0, A, B)
        SS[0] += -factor * Q_function(na-1, nb-1, 2, A, B)
        SS[0] *= phase
        # p - p pi
        phase = (-1.0)**(lbtemp + 1)
        SS[1] +=  0.5*factor * Q_function(na+1, nb-1, 0, A, B)

        SS[1] += -0.5*factor * Q_function(na-1, nb-1, 0, A, B)
        SS[1] +=     -factor * Q_function(na-1, nb-1, 1, A, B)
        SS[1] += -0.5*factor * Q_function(na-1, nb-1, 2, A, B)
        SS[1] *= phase
        return SS

    # s - d overlap
    if la == 2 and lb == 0: 
        phase = (-1.0)**lbtemp
        factor = 0.25*np.sqrt(5.0)
        SS[0] +=     -factor * Q_function(na  , nb, 0, A, B)

        SS[0] +=  3.0*factor * Q_function(na-2, nb, 0, A, B)
        SS[0] +=  6.0*factor * Q_function(na-2, nb, 1, A, B)
        SS[0] +=  3.0*factor * Q_function(na-2, nb, 2, A, B)
        SS[0] *= phase
        return SS

    # p - d overlap
    if la == 2 and lb == 1:
        # p - d sigma
        phase = (-1.0)**lbtemp
        factor = 0.25*np.sqrt(15.0)
        SS[0] +=     -factor * Q_function(na  , nb-1, 0, A, B)
        SS[0] +=      factor * Q_function(na  , nb-1, 1, A, B)

        SS[0] +=  3.0*factor * Q_function(na-2, nb-1, 0, A, B)
        SS[0] +=  3.0*factor * Q_function(na-2, nb-1, 1, A, B)
        SS[0] += -3.0*factor * Q_function(na-2, nb-1, 2, A, B)
        SS[0] += -3.0*factor * Q_function(na-2, nb-1, 3, A, B)
        SS[0] *= phase
        # p - d pi
        phase = (-1.0)**(lbtemp + 1)
        factor = 3.0*factor/np.sqrt(3.0)
        SS[1] +=      factor * Q_function(na  , nb-1, 0, A, B)
        SS[1] +=      factor * Q_function(na  , nb-1, 1, A, B)

        SS[1] +=     -factor * Q_function(na-2, nb-1, 0, A, B)
        SS[1] += -3.0*factor * Q_function(na-2, nb-1, 1, A, B)
        SS[1] += -3.0*factor * Q_function(na-2, nb-1, 2, A, B)
        SS[1] +=     -factor * Q_function(na-2, nb-1, 3, A, B)
        SS[1] *= phase
        return SS

    # d - d overlap
    if la == 2 and lb == 2:
        # d - d sigma
        phase = (-1.0)**lbtemp
        factor = 0.625
        SS[0] +=      factor * Q_function(na  , nb  , 0, A, B)
        SS[0] += -3.0*factor * Q_function(na  , nb-2, 0, A, B)
        SS[0] +=  6.0*factor * Q_function(na  , nb-2, 1, A, B)
        SS[0] += -3.0*factor * Q_function(na  , nb-2, 2, A, B)

        SS[0] += -3.0*factor * Q_function(na-2, nb  , 0, A, B)
        SS[0] += -6.0*factor * Q_function(na-2, nb  , 1, A, B)
        SS[0] += -3.0*factor * Q_function(na-2, nb  , 2, A, B)

        SS[0] +=  9.0*factor * Q_function(na-2, nb-2, 0, A, B)
        SS[0] +=-18.0*factor * Q_function(na-2, nb-2, 2, A, B)
        SS[0] +=  9.0*factor * Q_function(na-2, nb-2, 4, A, B)
        SS[0] *= phase
        # d - d pi
        phase = (-1.0)**(lbtemp + 1)
        factor = 6.0*factor
        SS[1] +=      factor * Q_function(na  , nb-2, 0, A, B)
        SS[1] +=     -factor * Q_function(na  , nb-2, 2, A, B)

        SS[1] +=     -factor * Q_function(na-2, nb-2, 0, A, B)
        SS[1] += -2.0*factor * Q_function(na-2, nb-2, 1, A, B)
        SS[1] +=  2.0*factor * Q_function(na-2, nb-2, 3, A, B)
        SS[1] +=      factor * Q_function(na-2, nb-2, 4, A, B)
        SS[1] *= phase
        # d - d delta
        phase = (-1.0)**(lbtemp + 2)
        SS[2] +=0.25*factor * Q_function(na+2, nb-2, 0, A, B)

        SS[2] +=-0.5*factor * Q_function(na  , nb-2, 0, A, B)
        SS[2] +=    -factor * Q_function(na  , nb-2, 1, A, B)
        SS[2] +=-0.5*factor * Q_function(na  , nb-2, 2, A, B)

        SS[2] +=0.25*factor * Q_function(na-2, nb-2, 0, A, B)
        SS[2] +=     factor * Q_function(na-2, nb-2, 1, A, B)
        SS[2] += 1.5*factor * Q_function(na-2, nb-2, 2, A, B)
        SS[2] +=     factor * Q_function(na-2, nb-2, 3, A, B)
        SS[2] +=0.25*factor * Q_function(na-2, nb-2, 4, A, B)
        SS[2] *= phase
        return SS



#def dovlp(na, la, S, zeta):
#    """
#    Derivatives of the overlap integral with respect to x, y, z.
#    """
#    if na == 1:
#        factor = 1.0/np.sqrt(3.0)
#        DS[0,0,0] = factor * S[0,0]



if __name__ == '__main__':
    #test()

    na = 3
    nb = 3
    la = 2
    lb = 2

    r = 2.0
    zeta = 1.3
    zetb = 1.6

    p = 0.5 * r *(zeta + zetb)
    t = (zeta - zetb) / (zeta + zetb)

    SS = basic_overlap(na, la, nb, lb, p, t)
    SS *= normalize(na, nb, p, t)

    minl = min(la, lb)
    for lam in range(minl+1):
        print('{0:10.3f} {1:10.3f} {2:20.6f}'.format(r, t, SS[lam]))
