import numpy as np
import pandas as pd
import scipy as scp
import csv
from tqdm import tqdm
import cmath
import json
from math import radians
import os

from JSONHelpers import TypeEncoder, as_complex

#input parameters are in SI units
def Solve2dSlice_ToJSON_dimensioned(directory, fr, B0, L, N, fp, ep, kmin, kmax, thetadegs, Nk, kzoffset, kyoffset):
    # ---------- Define Constants ----------
    e = 1.6e-19 #C - charge on electron
    m_e = 9.1094e-31 #kg - mass of electron
    mu_0 = 4e-7*np.pi #H/m - permeability of free space
    ep_0 = 8.854e-12 #F/m - permitivity of free space
    c = 3e8 #m/s - speed of light

    wr = 2*np.pi*fr
    wc = e*B0/m_e
    wc_n = wc/wr

    kmin_n = c*kmin/wr
    kmax_n = c*kmax/wr
    kzoffset_n = c*kzoffset/wr
    kyoffset_n = c*kyoffset/wr

    L_n = wr*L/c

    def wp_n(x):
        return fp(x)/fr

    Solve2dSlice_ToJSON_normalized(directory, fr, wc_n, L_n, N, wp_n, ep, kmin_n, kmax_n, thetadegs, Nk, kzoffset_n, kyoffset_n)

#input parameters are normalized to the reference freuqency fr (except for fr, which is in SI units)
def Solve2dSlice_ToJSON_normalized(directory, fr, wc_n, L_n, N, wp_n, ep, kmin_n, kmax_n, thetadegs, Nk, kzoffset_n, kyoffset_n):

    deltax_n = 2*L_n /N
    xlist_n = np.asarray([(-L_n + (i+0.5)*deltax_n) for i in range(0,N)]) #List of x points
    
    wplist_n = np.asarray([wp_n(xi) for xi in xlist_n])
    eplist = np.asarray([ep(xi) for xi in xlist_n])

    klist_n = np.linspace(kmin_n, kmax_n, Nk)

    # ---------- Run For Specified 2D Slice ----------
    evals_n_list = []
    avgs_n_list = []
    absAvgs_n_list = []
    stds_n_list = []
    absStds_n_list = []

    pbar = tqdm(total=Nk, desc=f"({kzoffset_n} kzn, {kyoffset_n} kyn, {thetadegs} deg Slice Computation", position=0)
    for k_n in klist_n:
        ky_n = k_n * np.sin(radians(thetadegs)) + kyoffset_n
        kz_n = k_n * np.cos(radians(thetadegs)) + kzoffset_n
        evals_n, avgs_n, absAvgs_n, stds_n, absStds_n = kPointSolveSort(ky_n, kz_n, N, wc_n, wplist_n, eplist, deltax_n, xlist_n)

        evals_n_list.append(evals_n)
        avgs_n_list.append(avgs_n)
        absAvgs_n_list.append(absAvgs_n)
        stds_n_list.append(stds_n)
        absStds_n_list.append(absStds_n)

        pbar.update(1)

    save_dict = {
        'fr':fr,
        'N':N,
        'L_n':L_n,
        'wc_n':wc_n,
        'deltax_n':deltax_n,
        'xlist_n':xlist_n,
        'wplist_n':wplist_n,
        'eplist':eplist,
        'kmin_n':kmin_n,
        'kmax_n':kmax_n,
        'thetadegs':thetadegs,
        'Nk':Nk,
        'kzoffset_n':kzoffset_n,
        'kyoffset_n':kyoffset_n,
        'klist_n':klist_n,
        'evals_n_list':evals_n_list,
        'avgs_n_list':avgs_n_list,
        'absAvgs_n_list':absAvgs_n_list,
        'stds_n_list':stds_n_list,
        'absStds_n_list':absStds_n_list
    }
    if not os.path.exists(directory):
        os.makedirs(directory)
    dict_file = directory + f'/{kzoffset_n}kzn_{kyoffset_n}kyn_{thetadegs}deg_SliceSolve.json'
    with open(dict_file, 'w') as f: 
        json.dump(save_dict, f, cls=TypeEncoder)

# ----- Helper Functions -----

def kPointSolveSort(ky_n, kz_n, N, wc_n, wplist_n, eplist, deltax_n, xlist_n):

    eigsys = np.linalg.eig(BuildMatrix(ky_n, kz_n, N, wc_n, wplist_n, eplist, deltax_n))

    #sort in order of increasing eigenvalue
    sorted_indices = eigsys[0].argsort()
    sorted_evals = eigsys[0][sorted_indices]
    sorted_evecs = eigsys[1][:,sorted_indices]

    #cutoff eigenvalues below 0
    idmin = np.abs(sorted_evals).argmin()
    cutsort_evals = sorted_evals[idmin:]
    cutsort_evecs = sorted_evecs[:,idmin:]

    avgs, absAvgs, stds, absStds = dataFromEvecs(cutsort_evecs, xlist_n, N)

    return cutsort_evals, avgs, absAvgs, stds, absStds

def dataFromEvecs(evecMatrix, xlist_n, N):

    listLen = evecMatrix.shape[1]
    avgs = [None] * listLen
    absAvgs = [None] * listLen
    stds = [None] * listLen
    absStds = [None] * listLen

    for i in range(listLen):
        Evec = np.transpose(evecMatrix[:,i]).tolist()[0]
        normVec = np.array([np.linalg.norm(Evec[9*j:9*(j+1)]) for j in range(N)])

        if normVec != list(np.zeros(len(normVec))):
            avgs[i] = np.average(xlist_n, weights=normVec)
            absAvgs[i] = np.average(np.abs(xlist_n), weights=normVec)
            #can't confirm that the stds are correct as written, just an initial typing
            stds[i] = np.sqrt(np.average((xlist_n-avgs[i])**2, weights=normVec))
            absStds[i] = np.sqrt(np.average((np.abs(xlist_n)-absAvgs[i])**2, weights=normVec))
        else:
            avgs[i] = 0
            absAvgs[i] = 0
            stds[i] = 0
            absStds[i] = 0
    
    return np.array(avgs), np.array(absAvgs), np.array(stds), np.array(absStds)

def BuildMatrix(ky, kz, N, wc, wplist, eplist, deltax):
    #All inputs in terms of dimensionless variables (assume _n)

    M = np.matrix(np.zeros((9*N, 9*N), dtype=np.complex128))
    
    for im in range(0, N):
        i = 9*im
        #uxi row
        M[i, i+1] = -1j*wc #uyi
        M[i, i+3] = -1j*wplist[im] #Exi
        
        #uyi row
        M[i+1, i] = 1j*wc #uxi
        M[i+1, i+4] = -1j*wplist[im] #Eyi
        
        #uzi row
        M[i+2, i+5] = -1j*wplist[im] #Ezi
    
        #Exi row
        M[i+3, i] = (1j*wplist[im])/(eplist[im]) #uxi
        M[i+3, (i-2)%(9*N)] = (kz)/(2*eplist[im]) #By(i-1/2)
        M[i+3, i+7] = (kz)/(2*eplist[im]) #By(i+1/2)
        M[i+3, (i-1)%(9*N)] = -(ky)/(2*eplist[im]) #Bz(i-1/2)
        M[i+3, i+8] = -(ky)/(2*eplist[im]) #Bz(i+1/2)
        
        #Eyi row
        M[i+4, i+1] = (1j*wplist[im])/(eplist[im]) #uyi
        M[i+4, (i-3)%(9*N)] = -(kz)/(2*eplist[im]) #Bx(i-1/2)
        M[i+4, i+6] = -(kz)/(2*eplist[im]) #Bx(i+1/2)
        M[i+4, i+8] = -(1j)/(deltax*eplist[im]) #Bz(i+1/2)
        M[i+4, (i-1)%(9*N)] = (1j)/(deltax*eplist[im]) #Bz(i-1/2)
        
        #Ezi row
        M[i+5, i+2] = (1j*wplist[im])/(eplist[im]) #uzi
        M[i+5, (i-3)%(9*N)] = (ky)/(2*eplist[im]) #Bx(i-1/2)
        M[i+5, i+6] = (ky)/(2*eplist[im]) #Bx(i+1/2)
        M[i+5, i+7] = (1j)/(deltax*eplist[im]) #By(i+1/2)
        M[i+5, (i-2)%(9*N)] = -(1j)/(deltax*eplist[im]) #By(i-1/2)
        
        #Bx(i+1/2) row
        M[i+6, i+5] = ky/2 #Ezi
        M[i+6, (i+14)%(9*N)] = ky/2 #Ez(i+1)
        M[i+6, i+4] = -kz/2 #Eyi
        M[i+6, (i+13)%(9*N)] = -kz/2 #Ey(i+1)
        
        #By(i+1/2) row
        M[i+7, i+3] = kz/2 #Exi
        M[i+7, (i+12)%(9*N)] = kz/2 #Ex(i+1)
        M[i+7, (i+14)%(9*N)] = 1j/deltax #Ez(i+1)
        M[i+7, i+5] = -1j/deltax #Ezi
        
        #Bz(i+1/2) row
        M[i+8, (i+13)%(9*N)] = -1j/deltax #Ey(i+1)
        M[i+8, i+4] = 1j/deltax #Eyi
        M[i+8, i+3] = -ky/2 #Exi
        M[i+8, (i+12)%(9*N)] = -ky/2 #Ex(i+1)

        #THIS HAS PERIODIC BOUNDARY CONDITIONS

    return M
