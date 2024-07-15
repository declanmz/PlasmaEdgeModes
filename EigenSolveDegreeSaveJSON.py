import numpy as np
import pandas as pd
import scipy as scp
import csv
from tqdm import tqdm
import cmath
import json

# TODO:
# - Put all variables into dictionary, store as JSON

# ---------- Define Constants ----------
e = 1.6e-19 #C - charge on electron
m_e = 9.1094e-31 #kg - mass of electron
al = m_e/e #define constant used in matrix
alin = 1/al #al inverse
mu_0 = 4e-7*np.pi #H/m - permeability of free space
ep_0 = 8.854e-12 #F/m - permitivity of free space
c = 3e8 #m/s - speed of light

# ---------- Setup Parameters ----------
fp0 = 10e9 #10GHz
B0 = 87e-3 #mT
L = 15e-3 * 2 #7.5 mm
N = 300

wp0 = 2*np.pi*fp0
wc = B0/al
fc = wc / (2 *np.pi)

wmin = 0.05 * wp0
wmax = 1.5 * wp0

# ---------- Plasma Density Profile ----------
Lscale = 4e-3 #1mm
qstart = 6.5e-3 *2 #mm
offset = qstart - 2.5e-3
qthickness = 1e-3 #1mm

def wp(x): #plasma frequency as a function of x
    if np.abs(x) >= qstart:
        return 0
    return wp0 * 0.5*(np.tanh((x+offset)/Lscale) - np.tanh((x-offset)/Lscale))

def ep(x): #relative permitivity of background medium (not including plasma)
    if np.abs(x) > qstart and np.abs(x) < qstart + qthickness:
        return 4
    return 1
    
def mu(x): #relative permeability of background medium (not including plasma)
    return 1

deltax = 2*L / N #Division betweeen points
xlist = [(-L + (i+0.5)*deltax) for i in range(0,N)] #List of x points

wplist = [wp(xi) for xi in xlist]
eplist = [ep(xi) for xi in xlist]
mulist = [mu(xi) for xi in xlist]


# ---------- ky kz Line at Specific Angle ----------
k0min = -4
k0max = 4
thetadegs = 38
Nk = 100
kzoffset = 0

k0list = np.linspace(k0min,k0max,Nk)
theta = thetadegs / 180 * np.pi

# ---------- Computation Functions ----------
def BuildMatrix(ky, kz):
    #L is length of x bounds [-L, L], in meters
    #N is the total number of points
    #ky, kz wavenumbers
    #B0 strength of magnetic field in T (in z-direction)
    #wp is plasma frequency as function of x
    #ep is background permitivity as function of x
    #mu is background permeability as function of x

    M = np.matrix(np.zeros((9*N, 9*N), dtype=np.complex128))
    
    for im in range(0, N):
        i = 9*im
        #vxi row
        M[i, i+3] = -1j*alin
        M[i, i+1] = -1j*alin*B0
        
        #vyi row
        M[i+1, i+4] = -1j*alin
        M[i+1, i] = 1j*alin*B0
        
        #vzi row
        M[i+2, i+5] = -1j*alin
    
        #Exi row
        M[i+3, (i-1)%(9*N)] = -(c**2 * ky)/(2*mulist[im]*eplist[im])
        M[i+3, i+8] = -(c**2 * ky)/(2*mulist[im]*eplist[im])
        M[i+3, (i-2)%(9*N)] = (c**2 * kz)/(2*mulist[im]*eplist[im])
        M[i+3, i+7] = (c**2 * kz)/(2*mulist[im]*eplist[im])
        M[i+3, i] = (1j*al*wplist[im]**2)/(eplist[im])
        
        #Eyi row
        M[i+4, (i-3)%(9*N)] = -(c**2 * kz)/(2*mulist[im]*eplist[im])
        M[i+4, i+6] = -(c**2 * kz)/(2*mulist[im]*eplist[im])
        M[i+4, i+8] = -(1j * c**2)/(deltax * mulist[im]*eplist[im])
        M[i+4, (i-1)%(9*N)] = (1j * c**2)/(deltax * mulist[im]*eplist[im])
        M[i+4, i+1] = (1j*al*wplist[im]**2)/(eplist[im])
        
        #Ezi row
        M[i+5, i+7] = (1j * c**2)/(deltax * mulist[im]*eplist[im])
        M[i+5, (i-2)%(9*N)] = -(1j * c**2)/(deltax * mulist[im]*eplist[im])
        M[i+5, (i-3)%(9*N)] = (c**2 * ky)/(2*mulist[im]*eplist[im])
        M[i+5, i+6] = (c**2 * ky)/(2*mulist[im]*eplist[im])
        M[i+5, i+2] = (1j*al*wplist[im]**2)/(eplist[im])
        
        #Bx(i+1/2) row
        M[i+6, i+5] = ky/2
        M[i+6, (i+14)%(9*N)] = ky/2
        M[i+6, i+4] = -kz/2
        M[i+6, (i+13)%(9*N)] = -kz/2
        
        #By(i+1/2) row
        M[i+7, i+3] = kz/2
        M[i+7, (i+12)%(9*N)] = kz/2
        M[i+7, (i+14)%(9*N)] = 1j/deltax
        M[i+7, i+5] = -1j/deltax
        
        #Bz(i+1/2) row
        M[i+8, (i+13)%(9*N)] = -1j/deltax
        M[i+8, i+4] = 1j/deltax
        M[i+8, i+3] = -ky/2
        M[i+8, (i+12)%(9*N)] = -ky/2

        #RIGHT NOW THIS HAS PERIODIC BOUNDARY CONDITIONS

    return M

def EAvgStdMaxList(evecMatrix):
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

def CutSortedEigensolve(k0mag, thetadegs, kzoffset, wmin, wmax, directory):
    kmag = wp0 * k0mag / c
    ky = kmag * np.sin(theta)
    kz = kmag * np.cos(theta) + kzoffset

    eigsys = np.linalg.eig(BuildMatrix(ky, kz))

    #sort in order of increasing eigenvalue
    sorted_indices = eigsys[0].argsort()
    sorted_evals = eigsys[0][sorted_indices]
    sorted_evecs = eigsys[1][:,sorted_indices]

    #cutoff eigenvalues below wmin and above wmax
    idmin = np.abs(sorted_evals - wmin).argmin()
    idmax = np.abs(sorted_evals - wmax).argmin()
    cutsort_evals = sorted_evals[idmin:idmax]
    cutsort_evecs = sorted_evecs[:,idmin:idmax]

    #find Emax, Eavg, and Estd
    cutsort_Eavg, cutsort_Estd, cutsort_Emax = EAvgStdMaxList(cutsort_evecs)

    #save evecs to json
    save_evecs = cutsort_evecs.tolist()
    file = directory + '/EvecData_CutSort/' + f'{k0mag:.5f}'.replace('.','_') + f'k0_{thetadegs}deg.json' #can't have decimal point in file name
    with open(file, 'w') as f: 
        json.dump(save_evecs, f)

    return cutsort_evals, cutsort_Eavg, cutsort_Estd, cutsort_Emax



# ---------- Run For Specific Degree ----------
directory = './SetupOG'

evals_list = []
Eavg_list = []
Estd_list = []
Emax_list = []

pbar = tqdm(total=Nk, desc=f"{thetadegs}deg Computation", position=0)
for k0 in k0list:
    evals, Eavg, Estd, Emax = CutSortedEigensolve(k0, thetadegs, kzoffset, wmin, wmax, directory)

    evals_list.append(evals)
    Eavg_list.append(Eavg)
    Estd_list.append(Estd)
    Emax_list.append(Emax)

    pbar.update(1)

print('Saving... ')
save_dict = {
    'fp0':fp0,
    'wp0':wp0,
    'B0':B0,
    'N':N,
    'L':L,
    'wc':wc,
    'fc':fc,
    'wmin':wmin,
    'wmax':wmax,
    'deltax':deltax,
    'xlist':xlist,
    'wplist':wplist,
    'eplist':eplist,
    'mulist':mulist,
    'k0min':k0min,
    'k0max':k0max,
    'thetadegs':thetadegs,
    'theta':theta,
    'Nk':Nk,
    'kzoffset':kzoffset,
    'k0list':k0list,
    'evals_list':evals_list,
    'Eavg_list':Eavg_list,
    'Estd_list':Estd_list,
    'Emax_list':Emax_list
}

dict_file = directory + f'/{thetadegs}deg_CutSort'
with open(dict_file, 'w') as f: 
    json.dump(save_dict, f)

print('Done.')
#next is to write the code to run this function over all k0 for a certain degree, then save json dictionary

# the cutsort program should be run, then all the data should be stored raw (in one Json per single angle k0 sweep)
# For a single setup (i.e. plasma profile, density, length, etc), all files from that setup should be in one folder (say SetupA)
# Then inside that folder there is the EvecData_CutSort folder, then a bunch of cutsort jsons for degrees from -38 to +38
# then there will be another program I can run to generate filtered jsons from the cutsort jsons
# then there will be a program to run plotting from these jsons (and I can swap out file names easily)
# then there will be a program to find the intersections with the microstrip dispersion - and then plot them