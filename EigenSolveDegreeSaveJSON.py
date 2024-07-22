import numpy as np
import pandas as pd
import scipy as scp
import csv
from tqdm import tqdm
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex

# ---------- Computation Functions ----------
def BuildMatrix(ky, kz, N, wc, wplist, eplist, deltax):
    #All inputs in terms of dimensionless variables (assume _n)

    M = np.matrix(np.zeros((9*N, 9*N), dtype=np.complex128))
    
    for im in range(0, N):
        i = 9*im
        #uxi row
        M[i, i+1] = -1j*wc #uyi
        M[i, i+3] = -1j #Exi
        
        #uyi row
        M[i+1, i] = 1j*wc #uxi
        M[i+1, i+4] = -1j #Eyi
        
        #uzi row
        M[i+2, i+5] = -1j #Ezi
    
        #Exi row
        M[i+3, i] = (1j*wplist[im]**2)/(eplist[im]) #uxi
        M[i+3, (i-2)%(9*N)] = (kz)/(2*eplist[im]) #By(i-1/2)
        M[i+3, i+7] = (kz)/(2*eplist[im]) #By(i+1/2)
        M[i+3, (i-1)%(9*N)] = -(ky)/(2*eplist[im]) #Bz(i-1/2)
        M[i+3, i+8] = -(ky)/(2*eplist[im]) #Bz(i+1/2)
        
        #Eyi row
        M[i+4, i+1] = (1j*wplist[im]**2)/(eplist[im]) #uyi
        M[i+4, (i-3)%(9*N)] = -(kz)/(2*eplist[im]) #Bx(i-1/2)
        M[i+4, i+6] = -(kz)/(2*eplist[im]) #Bx(i+1/2)
        M[i+4, i+8] = -(1j)/(deltax*eplist[im]) #Bz(i+1/2)
        M[i+4, (i-1)%(9*N)] = (1j)/(deltax*eplist[im]) #Bz(i-1/2)
        
        #Ezi row
        M[i+5, i+2] = (1j*wplist[im]**2)/(eplist[im]) #uzi
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

def EAvgStdMaxList(evecMatrix, xlist, N):
    #All inputs in terms of dimensionless variables (assume _n)
    listLen = evecMatrix.shape[1]
    Eavg = [None] * listLen
    Estd = [None] * listLen
    Emax = [None] * listLen
    for i in range(listLen):
        Evec = np.transpose(evecMatrix[:,i]).tolist()[0]
        Ex = [Evec[9*j + 3] for j in range(N)]
        Ey = [Evec[9*j + 4] for j in range(N)]
        Ez = [Evec[9*j + 5] for j in range(N)]
        
        Emag = [np.sqrt(np.abs(Ex[j])**2 + np.abs(Ey[j])**2 + np.abs(Ez[j])**2) for j in range(N)]
        Eavg[i] = np.average(xlist, weights=Emag)
        Estd[i] = np.sqrt(np.average(np.multiply(xlist, xlist), weights=Emag) - Eavg[i]**2)
        Emax[i] = np.array([np.abs(Ex).max(), np.abs(Ey).max(), np.abs(Ez).max()]).max()
    
    return np.array(Eavg), np.array(Estd), np.array(Emax)

def CutSortedEigensolve(kmag, kzoffset, wmin, wmax, theta, N, wc, wplist, eplist, deltax, xlist):
    #All inputs in terms of dimensionless variables (assume _n)
    pbarInner = tqdm(total=4, desc=f"Eigensolve {kmag}", position=1, leave=False)
    ky = kmag * np.sin(theta)
    kz = kmag * np.cos(theta) + kzoffset

    eigsys = np.linalg.eig(BuildMatrix(ky, kz, N, wc, wplist, eplist, deltax))
    pbarInner.update(1)

    #sort in order of increasing eigenvalue
    sorted_indices = eigsys[0].argsort()
    sorted_evals = eigsys[0][sorted_indices]
    sorted_evecs = eigsys[1][:,sorted_indices]
    pbarInner.update(1)

    #cutoff eigenvalues below wmin and above wmax
    idmin = np.abs(sorted_evals - wmin).argmin()
    idmax = np.abs(sorted_evals - wmax).argmin()
    cutsort_evals = sorted_evals[idmin:idmax]
    cutsort_evecs = sorted_evecs[:,idmin:idmax]
    pbarInner.update(1)

    #find Emax, Eavg, and Estd
    cutsort_Eavg, cutsort_Estd, cutsort_Emax = EAvgStdMaxList(cutsort_evecs, xlist, N)
    pbarInner.update(1)

    #save evecs to json
    # save_evecs = cutsort_evecs.tolist()
    # pbarInner.update(1)
    # file = directory + '/EvecData_CutSort/evec_' + f'{k0mag:.5f}'.replace('.','_') + f'k0_{thetadegs}deg.json' #can't have decimal point in file name
    # with open(file, 'w+') as f: 
    #     json.dump(save_evecs, f, cls=ComplexEncoder)
    # pbarInner.update(1)

    return cutsort_evals, cutsort_Eavg, cutsort_Estd, cutsort_Emax

# ---------- main function ----------
def EigenSolveDegree_SaveJSON(directory, fr, B0, L, N, fmin, fmax, wp, ep, kmin, kmax, thetadegs, Nk, kzoffset, fp0):

    #All normalized variables writen as x_n, all dimensioned variables written as normal

    # ---------- Define Constants ----------
    e = 1.6e-19 #C - charge on electron
    m_e = 9.1094e-31 #kg - mass of electron
    mu_0 = 4e-7*np.pi #H/m - permeability of free space
    ep_0 = 8.854e-12 #F/m - permitivity of free space
    c = 3e8 #m/s - speed of light

    # ---------- Setup Parameters ----------
    wr = 2*np.pi*fr
    wc = e*B0/m_e
    fc = wc / (2 *np.pi)
    wc_n = wc/wr
    wmin = 2*np.pi*fmin
    wmax = 2*np.pi*fmax
    wmin_n = wmin/wr
    wmax_n = wmax/wr

    kmin_n = c*kmin/wr
    kmax_n = c*kmax/wr
    kzoffset_n = c*kzoffset/wr

    L_n = wr*L/c
    deltax = 2*L / N #Division betweeen points
    deltax_n = wr*deltax/c
    xlist = np.asarray([(-L + (i+0.5)*deltax) for i in range(0,N)]) #List of x points
    xlist_n = wr*xlist/c
    
    wplist = np.asarray([wp(xi) for xi in xlist])
    wplist_n = wplist/wr
    eplist = np.asarray([ep(xi) for xi in xlist])

    klist_n = np.linspace(kmin_n, kmax_n, Nk)
    theta = thetadegs / 180 * np.pi

    # ---------- Run For Specific Degree ----------
    evals_list_n = []
    Eavg_list_n = []
    Estd_list_n = []
    Emax_list_n = []

    pbar = tqdm(total=Nk, desc=f"{thetadegs}deg Computation", position=0)
    for k_n in klist_n:
        evals_n, Eavg_n, Estd_n, Emax_n = CutSortedEigensolve(k_n, kzoffset_n, wmin_n, wmax_n, theta, N, wc_n, wplist_n, eplist, deltax_n, xlist_n)

        evals_list_n.append(evals_n)
        Eavg_list_n.append(Eavg_n)
        Estd_list_n.append(Estd_n)
        Emax_list_n.append(Emax_n)

        pbar.update(1)

    save_dict = {
        'fp0':fp0,
        'wp0':2*np.pi*fp0,
        'fr':fr,
        'wr':wr,
        'B0':B0,
        'N':N,
        'L':L,
        'L_n':L_n,
        'wc':wc,
        'wc_n':wc_n,
        'fc':fc,
        'wmin':wmin,
        'wmin_n':wmin_n,
        'wmax':wmax,
        'wmax_n':wmax_n,
        'deltax':deltax,
        'deltax_n':deltax_n,
        'xlist':xlist,
        'xlist_n':xlist_n,
        'wplist':wplist,
        'wplist_n':wplist_n,
        'eplist':eplist,
        'kmin':kmin,
        'kmin_n':kmin_n,
        'kmax':kmax,
        'kmax_n':kmax_n,
        'thetadegs':thetadegs,
        'theta':theta,
        'Nk':Nk,
        'kzoffset':kzoffset,
        'kzoffset_n':kzoffset_n,
        'klist_n':klist_n,
        'evals_list_n':evals_list_n,
        'Eavg_list_n':Eavg_list_n,
        'Estd_list_n':Estd_list_n,
        'Emax_list_n':Emax_list_n
    }

    dict_file = directory + f'/{thetadegs}deg_Unfiltered.json'
    with open(dict_file, 'w') as f: 
        json.dump(save_dict, f, cls=TypeEncoder)

# ---------- Setup parameters ----------
scaling = 1

fp0 = 10e9 #10GHz
B0 = 87e-3 #mT
L = 15e-3 * scaling
N = 100 #300 for good results
fr = 1e9 #1 GHz
fmin = 0.5 * fr
fmax = 15 * fr

wp0 = 2*np.pi*fp0

# ---------- Plasma Density Profile ----------
Lscale = 2e-3 * scaling#1mm
qstart = 6.5e-3 * scaling #mm
offset = qstart - 1.25e-3 * scaling
qthickness = 1e-3 #1mm

def wp(x): #plasma frequency as a function of x
    if np.abs(x) >= qstart:
        return 0
    return wp0*0.5*(np.tanh((x+offset)/Lscale) - np.tanh((x-offset)/Lscale))

# def wp(x): #quadratic density profile
#     if np.abs(x) >= qstart:
#         return 0
#     return (-wp0/(qstart**2))*(x-qstart)*(x+qstart)

def ep(x): #relative permitivity of background medium (not including plasma)
    if np.abs(x) > qstart and np.abs(x) < qstart + qthickness:
        return 4
    return 1

# ---------- ky kz Line at Specific Angle ----------
kmin = -500 #1/m
kmax = 500 #1/m
Nk = 20 #100 for good results
kzoffset = 0

directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/Testing'
thetadegs = 39

# EigenSolveDegree_SaveJSON(directory, fp0, B0, L, N, w0min, w0max, wp, ep, mu, k0min, k0max, thetadegs, Nk, kzoffset)
EigenSolveDegree_SaveJSON(directory, fr, B0, L, N, fmin, fmax, wp, ep, kmin, kmax, thetadegs, Nk, kzoffset, fp0)