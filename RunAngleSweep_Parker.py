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

n0 = 4e11 * 1e6
wp0 = np.sqrt((n0 * e**2)/(m_e * ep_0))
fp0 = wp0/(2*np.pi) 
print(fp0)
B0 = 0.1 #T
L = 50e-2 * sizeScaling
N = 200 #300
fr = 1e9 #1GHz
fmin = 0 * fr #0.5 * fr
fmax = 1.5 * fp0
# wp0 = 2*np.pi*fp0

# ---------- Plasma Density Profile ----------
# Lscale = 2e-3 #1mm
# qstart = 6.5e-3  #mm
offset = 25e-2
# qthickness = 1e-3 #1mm

Lscale = 5e-2 * sizeScaling #1mm
# qstart = 6.5e-3 * sizeScaling  #mm
# offset = qstart - (2.5e-3 * sizeScaling)
# qthickness = 1e-3 * sizeScaling #1mm

# def wp(x): #quadratic density profile
#     if np.abs(x) >= qstart:
#         return 0
#     return (-wp0/(qstart**2))*(x-qstart)*(x+qstart)

def wp(x): #plasma frequency as a function of x
    return wp0*0.5*(np.tanh((x+offset)/Lscale) - np.tanh((x-offset)/Lscale))

def ep(x): #relative permitivity of background medium (not including plasma)
    return 1

# ---------- ky kz Line at Specific Angle ----------
kmin = -500 #1/m
kmax = 500 #1/m
Nk = 100 #100
kzoffset = 0.8 * wp0 / c


directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/ParkerReproduce'
thetadegsList = np.arange(0,40,2) #2

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
EigenSolveDegree_SaveJSON(directory, fr, B0, L, N, fmin, fmax, wp, ep, kmin, kmax, thetadegs, Nk, kzoffset, fp0)
UnfilteredFile = directory + f'/{thetadegs}deg_Unfiltered.json'
findWgIntersection_UpdateJSON(UnfilteredFile, wgEpsilon, thetadegs)
filterEigJSON(UnfilteredFile, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs)
Filteredfile = directory + f'/{thetadegs}deg_{filterName}.json'
findWgIntersection_UpdateJSON(Filteredfile, wgEpsilon, thetadegs)


print('Angle Sweep Finished.')