import numpy as np

from PlasmaEdgeModes.V1.EigenSolveDegreeSaveJSON import EigenSolveDegree_SaveJSON
from PlasmaEdgeModes.V1.EigenJsonFiltering import filterEigJSON
from WaveguideIntersectionToJSON import findWgIntersection_UpdateJSON

# ---------- Define Constants ----------
e = 1.6e-19 #C - charge on electron
m_e = 9.1094e-31 #kg - mass of electron
al = m_e/e #define constant used in matrix
alin = 1/al #al inverse
mu_0 = 4e-7*np.pi #H/m - permeability of free space
ep_0 = 8.854e-12 #F/m - permitivity of free space
c = 3e8 #m/s - speed of light

# ---------- Setup parameters ----------
# NoQuartzA has no quartz, a smooth plasma profile,
# lower plasma frequency of 6 GHz, and the actually correct length

fp0 = 6e9 #10GHz
B0 = 87e-3 #mT
L = 15e-3
N = 300 #300
w0min = 0.05
w0max = 1.5

wp0 = 2*np.pi*fp0

# ---------- Plasma Density Profile ----------
Lscale = 2e-3 #mm
# qstart = 6.5e-3 *2 #mm
# offset = qstart - 2.5e-3
# qthickness = 1e-3 #1mm
offset = 7.5e-3

def wp(x): #plasma frequency as a function of x
    return wp0*0.5*(np.tanh((x+offset)/Lscale) - np.tanh((x-offset)/Lscale))

def ep(x): #relative permitivity of background medium (not including plasma)
    return 1
    
def mu(x): #relative permeability of background medium (not including plasma)
    return 1

# ---------- ky kz Line at Specific Angle ----------
k0min = -2.5
k0max = 2.5
Nk = 100 #100
kzoffset = 0


directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/Setup_NoQuartzA'
thetadegsList = np.arange(0,40,2) #2

filterName = 'FilterA'
filter_posAvgBound = 15e-3
filter_negAvgBound = -2.5e-3
filter_maxStd = 1 #4e-3
filter_EmaxMin = 1e-5

wgEpsilon = 1e9

for thetadegs in thetadegsList:
    EigenSolveDegree_SaveJSON(directory, fp0, B0, L, N, w0min, w0max, wp, ep, mu, k0min, k0max, thetadegs, Nk, kzoffset)
    CutSortfile = directory + f'/{thetadegs}deg_CutSort.json'
    filterEigJSON(CutSortfile, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs)
    Filteredfile = directory + f'/{thetadegs}deg_{filterName}.json'
    findWgIntersection_UpdateJSON(Filteredfile, wgEpsilon, thetadegs)

print('Angle Sweep Finished.')