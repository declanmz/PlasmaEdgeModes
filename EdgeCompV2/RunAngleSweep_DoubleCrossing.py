import numpy as np

from EigenSolveDegreeSaveJSON import EigenSolveDegree_SaveJSON
from EigenJsonFiltering import filterEigJSON
from WaveguideIntersectionToJSON import findWgIntersection_UpdateJSON

# ---------- Define Constants ----------
e = 1.6e-19 #C - charge on electron
m_e = 9.1094e-31 #kg - mass of electron
mu_0 = 4e-7*np.pi #H/m - permeability of free space
ep_0 = 8.854e-12 #F/m - permitivity of free space
c = 3e8 #m/s - speed of light

# ---------- Setup parameters ----------
sizeScaling = 1
fr = 1e9 #5GHz
fp1 = 10*fr
fp2 = 0.2*fr
fc = 1.25*fr

B0 = (2*np.pi*fc*m_e)/e

L = 50e-2 * sizeScaling
N = 200 #300
fmin = 0 * fr #0.5 * fr
fmax = 1.5 * fp1


wp1 = 2*np.pi*fp1
wp2 = 2*np.pi*fp2

# ---------- Plasma Density Profile ----------
# Lscale = 2e-3 #1mm
# qstart = 6.5e-3  #mm
# offset = qstart - 1.25e-3
# qthickness = 1e-3 #1mm

Lscale = 5e-2 * sizeScaling #1mm
offset = L/2

# def wp(x): #quadratic density profile
#     if np.abs(x) >= qstart:
#         return 0
#     return (-wp0/(qstart**2))*(x-qstart)*(x+qstart)

def wp(x): #plasma frequency as a function of x
    return wp1
    # return (wp1-wp2)*0.5*(np.tanh((x+offset)/Lscale) - np.tanh((x-offset)/Lscale)) + wp2

def ep(x): #relative permitivity of background medium (not including plasma)
    return 1

# ---------- ky kz Line at Specific Angle ----------
kmin = -500 #1/m
kmax = 500 #1/m
Nk = 50 #100
kzoffset = 1.125 * (2*np.pi*fr)/c


directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/FlatFrequency'

filterName = 'FilterRight'
filter_posAvgBound = 50e-2 * sizeScaling
filter_negAvgBound = 0 * sizeScaling
filter_maxStd = 1 * sizeScaling #4e-3
filter_EmaxMin = 1e-5

wgEpsilon = 0.1

# for thetadegs in thetadegsList:
#     EigenSolveDegree_SaveJSON(directory, fr, B0, L, N, fmin, fmax, wp, ep, kmin, kmax, thetadegs, Nk, kzoffset, fp0)
#     UnfilteredFile = directory + f'/{thetadegs}deg_Unfiltered.json'
#     findWgIntersection_UpdateJSON(UnfilteredFile, wgEpsilon, thetadegs)
#     filterEigJSON(UnfilteredFile, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs)
#     Filteredfile = directory + f'/{thetadegs}deg_{filterName}.json'
#     findWgIntersection_UpdateJSON(Filteredfile, wgEpsilon, thetadegs)

# for thetadegs in thetadegsList:
#     UnfilteredFile = directory + f'/{thetadegs}deg_Unfiltered.json'
#     filterEigJSON(UnfilteredFile, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs)
#     Filteredfile = directory + f'/{thetadegs}deg_{filterName}.json'
#     findWgIntersection_UpdateJSON(Filteredfile, wgEpsilon, thetadegs)


thetadegs = 90
EigenSolveDegree_SaveJSON(directory, fr, B0, L, N, fmin, fmax, wp, ep, kmin, kmax, thetadegs, Nk, kzoffset, fp1)
UnfilteredFile = directory + f'/{thetadegs}deg_Unfiltered.json'
findWgIntersection_UpdateJSON(UnfilteredFile, wgEpsilon, thetadegs)
filterEigJSON(UnfilteredFile, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs)
Filteredfile = directory + f'/{thetadegs}deg_{filterName}.json'
findWgIntersection_UpdateJSON(Filteredfile, wgEpsilon, thetadegs)


print('Angle Sweep Finished.')