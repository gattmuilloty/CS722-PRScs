#!/usr/bin/env python

"""
Parse the reference panel
"""

import numpy as np
import h5py
from scipy import linalg
import math
from numpy import random

def parse_ldblk(ldblk_dir, sst_df, chrom):

    # Define file path for HDF5 based on ldblk_dir
    chr_name = ldblk_dir + '/ldblk_1kg_chr' + str(chrom) + '.hdf5'

    # Open the HDF5 file
    hdf_chr = h5py.File(chr_name, 'r')
    n_blk = len(hdf_chr)
    
    # Reading LD blocks
    ld_blk = [np.array(hdf_chr['blk_'+str(blk)]['ldblk']) for blk in range(1, n_blk+1)]

    snp_blk = []
    for blk in range(1, n_blk+1):
        snp_blk.append([bb.decode("UTF-8") for bb in list(hdf_chr['blk_'+str(blk)]['snplist'])])

    blk_size = []
    mm = 0
    for blk in range(n_blk):
        idx = [ii for (ii, snp) in enumerate(snp_blk[blk]) if snp in sst_df['SNP'].values]
        blk_size.append(len(idx))
        if idx:
            idx_blk = range(mm, mm + len(idx))
            flip = sst_df.loc[idx_blk, 'FLP'].values
            ld_blk[blk] = ld_blk[blk][np.ix_(idx, idx)] * np.outer(flip, flip)

            _, s, v = linalg.svd(ld_blk[blk])
            h = np.dot(v.T, np.dot(np.diag(s), v))
            ld_blk[blk] = (ld_blk[blk] + h) / 2

            mm += len(idx)
        else:
            ld_blk[blk] = np.array([])

    return ld_blk, blk_size



"""
Random variate generator for the generalized inverse Gaussian distribution.
Reference: L Devroye. Random variate generation for the generalized inverse Gaussian distribution.
           Statistics and Computing, 24(2):239-246, 2014.
"""

def psi(x, alpha, lam):
    f = -alpha*(math.cosh(x)-1.0)-lam*(math.exp(x)-x-1.0)
    return f


def dpsi(x, alpha, lam):
    f = -alpha*math.sinh(x)-lam*(math.exp(x)-1.0)
    return f


def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return f


def gigrnd(p, a, b):
    # setup -- sample from the two-parameter version gig(lam,omega)
    p = float(p); a = float(a); b = float(b)
    lam = p
    omega = math.sqrt(a*b)

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(math.pow(omega,2)+math.pow(lam,2))-lam

    # find t
    x = -psi(1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.sqrt(2.0/(alpha+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.log(4.0/(alpha+2.0*lam))

    # find s
    x = -psi(-1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            s = math.sqrt(4.0/(alpha*math.cosh(1)+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0/lam
        elif lam == 0:
            s = math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha))
        else:
            s = min(1.0/lam, math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha)))

    # find auxiliary parameters
    eta = -psi(t, alpha, lam)
    zeta = -dpsi(t, alpha, lam)
    theta = -psi(-s, alpha, lam)
    xi = dpsi(-s, alpha, lam)

    p = 1.0/xi
    r = 1.0/zeta

    td = t-r*eta
    sd = s-p*theta
    q = td+sd

    # random variate generation
    while True:
        U = random.random()
        V = random.random()
        W = random.random()
        if U < q/(p+q+r):
            rnd = -sd+q*V
        elif U < (q+r)/(p+q+r):
            rnd = td-r*math.log(V)
        else:
            rnd = -sd+p*math.log(V)

        f1 = math.exp(-eta-zeta*(rnd-t))
        f2 = math.exp(-theta+xi*(rnd+s))
        if W*g(rnd, sd, td, f1, f2) <= math.exp(psi(rnd, alpha, lam)):
            break

    # transform back to the three-parameter version gig(p,a,b)
    rnd = math.exp(rnd)*(lam/omega+math.sqrt(1.0+math.pow(lam,2)/math.pow(omega,2)))
    if swap:
        rnd = 1.0/rnd

    rnd = rnd/math.sqrt(a/b)
    return rnd