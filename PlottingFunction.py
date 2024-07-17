#region ----- Import Statements -----
import numpy as np
import pandas as pd
import scipy as scp

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib as mpl
import csv
from tqdm import notebook
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex
from EvecSparseEigensolve import EvecSparseEigensolve_toJSON
#endregion

#region ----- Define Constants -----
e = 1.6e-19 #C - charge on electron
m_e = 9.1094e-31 #kg - mass of electron
al = m_e/e #used in matrix
alin = 1/al
mu_0 = 4e-7*np.pi #H/m - permeability of free space
ep_0 = 8.854e-12 #F/m - permitivity of free space
c = 3e8 #m/s - speed of light
#endregion

# ----- Choose File -----
baseDirectory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes'
setupFolder = 'SetupOG_ReducedRange'
filterName = 'FilterA'
thetadegs = 38

#region ----- Load JSON Data -----
file = f'{baseDirectory}/{setupFolder}/{thetadegs}deg_{filterName}.json'
with open(file, 'r') as f:
    jsondata = json.load(f, object_hook=as_complex)
# unpacking:
N = jsondata['N']
L = jsondata['L']
deltax = jsondata['deltax']
wp0 = jsondata['wp0']
thetadegs = jsondata['thetadegs']
theta = jsondata['theta']
Nk = jsondata['Nk']
wmin = jsondata['wmin']
wmax = jsondata['wmax']
k0min = jsondata['k0min']
k0max = jsondata['k0max']
B0 = jsondata['B0']
wc = jsondata['wc']
kzoffset = jsondata['kzoffset']
fp0 = jsondata['fp0']

xlist = jsondata['xlist']
wplist = jsondata['wplist']
eplist = jsondata['eplist']
mulist = jsondata['mulist']

k0list = np.asarray(jsondata['k0list'])
evals_list = list(map(np.array, jsondata['evals_list']))
Eavg_list = list(map(np.array, jsondata['Eavg_list']))
Estd_list = list(map(np.array, jsondata['Estd_list']))
Emax_list = list(map(np.array, jsondata['Emax_list']))

try:
    wgNeg_k0list = np.asarray(jsondata['wgNeg_k0list'])
    wgNeg_distanceList = np.asarray(jsondata['wgNeg_distanceList'])
    wgNeg_evalList = np.asarray(jsondata['wgNeg_evalList'])
    wgNeg_EavgList = jsondata['wgNeg_EavgList']
    wgNeg_EstdList = jsondata['wgNeg_EstdList']
    wgNeg_EmaxList = jsondata['wgNeg_EmaxList']

    wgPos_k0list = np.asarray(jsondata['wgPos_k0list'])
    wgPos_distanceList = np.asarray(jsondata['wgPos_distanceList'])
    wgPos_evalList = np.asarray(jsondata['wgPos_evalList'])
    wgPos_EavgList = jsondata['wgPos_EavgList']
    wgPos_EstdList = jsondata['wgPos_EstdList']
    wgPos_EmaxList = jsondata['wgPos_EmaxList']
except:
    print("No waveguide data")
#endregion

#region ----- Bulk Modes Calculation -----
def MagPlasmaEigenmodes(sig, kn_x, kn_y, kn_z):
    H = np.array([[0,-sig*1j,0,-1j,0,0,0,0,0],
                  [sig*1j,0,0,0,-1j,0,0,0,0],
                  [0,0,0,0,0,-1j,0,0,0],
                  [1j,0,0,0,0,0,0,kn_z,-kn_y],
                  [0,1j,0,0,0,0,-kn_z,0,kn_x],
                  [0,0,1j,0,0,0,kn_y,-kn_x,0],
                  [0,0,0,0,-kn_z,kn_y,0,0,0],
                  [0,0,0,kn_z,0,-kn_x,0,0,0],
                  [0,0,0,-kn_y,kn_x,0,0,0,0]])
    omn, fn = np.linalg.eig(H)
    #fn = (vn, En, Bn), 9x1 vector
    return [omn.real, fn]

dots = 1000
k0line = np.linspace(k0min,k0max,dots)
ky0line = k0line * np.sin(theta)
kz0line = k0line * np.cos(theta) + kzoffset
bulkLines = np.empty((5,dots))
sig = wc / wp0

for n in range(dots):
    omns = np.sort(MagPlasmaEigenmodes(sig, ky0line[n], 0, kz0line[n])[0])
    for i in range(5):
        bulkLines[i][n] = omns[i+4]

lightline = [np.sqrt(ky0line[i]**2 + kz0line[i]**2) for i in range(dots)]
#endregion

#region ----- Load Waveguide Dispersion -----
dwg = 6.25e-3
# wpwg = 2 * np.pi * 3.5e9
wpwg = wp0
kpwg = wpwg/c

dispersionload = np.loadtxt(f"{baseDirectory}/WaveguideDispersion/eps_4.txt")
k0waveguide = dispersionload[:,0]/180 /dwg * np.pi / kpwg
w0waveguide = dispersionload[:,1] * 2 * np.pi * 1e9 / wpwg
#endregion

# ----- Choose Preset for Layout -----
plotPreset = 1

#region --- Preset Variables ---
plotDispersion = False
plotBulkModes = False
plotLightLine = False
plotWgLine = False
plotWgIntersection = False

plotTextInfo = False
#endregion

if plotPreset == 1: #Dispersion Plot with Text Info
    fig, axs = plt.subplot_mosaic([['e', 'e', 'e', 'e', 't']], figsize=(12, 7))

    plotDispersion = True
    plotBulkModes = True
    plotLightLine = True
    plotWgLine = True
    plotWgIntersection = True

    plotTextInfo = True
    

if plotDispersion: #Properties of the Dispersion Plot + Plotting Itself
    # --- Dispersion Plotting Properties ---
    axs['e'].set_title('Eigenvalues')
    axs['e'].set_xlabel(r'$c k / \omega_{p0}$', size=12)
    axs['e'].set_ylabel(r'$\omega / \omega_{p0}$', size=12)
    axs['e'].grid(color='gray', linestyle='dashed', alpha=0.2)

    overwrite_xlim = [None, None] #Values in terms of k0
    overwrite_ylim = [None, None] #Values in terms of wp0

    dotsize = 10

    dispCmap = 'coolwarm'
    dispColorbarOrientation = 'vertical'
    dispCmap_case = 'Eavg'
    match dispCmap_case:
        case 'Eavg':
            dispCmap_values = Eavg_list
            dispCmap_norm = plt.Normalize(-5e-3, 15e-3)
            dispColorbarLabel = 'E Field Centroid along x-axis [m]'
        case _:
            Exception("Improper Dispersion Cmap Case")

    #region --- Dispersion Plotting Backend ---
    for n in range(len(k0list)):
        dispScatter = axs['e'].scatter([k0list[n] for i in evals_list[n]], evals_list[n].real / wp0, s=dotsize, 
                        c=dispCmap_values[n], norm=dispCmap_norm, cmap=dispCmap)
    
    if overwrite_xlim[0] == None:
        overwrite_xlim[0] = k0list[0]
    if overwrite_xlim[1] == None:
        overwrite_xlim[1] = k0list[-1]
    if overwrite_ylim[0] == None:
        overwrite_ylim[0] = wmin/wp0
    if overwrite_ylim[1] == None:
        overwrite_ylim[1] = wmax/wp0
    axs['e'].set_xlim(overwrite_xlim)
    axs['e'].set_ylim(overwrite_ylim)

    dispCbar = fig.colorbar(dispScatter, ax=axs['e'], orientation=dispColorbarOrientation)
    dispCbar.set_label(dispColorbarLabel)
    #endregion

    if plotBulkModes:
        for i in range(len(bulkLines)):
            axs['e'].plot(k0line, bulkLines[i], color='black', alpha=0.5)

    if plotLightLine:
        axs['e'].plot(k0line, lightline, color='yellow')

    if plotWgLine:
        wgLineColor = 'green'
        axs['e'].plot(k0waveguide, w0waveguide, color=wgLineColor)
        axs['e'].plot(-k0waveguide, w0waveguide, color=wgLineColor)

    if plotWgIntersection:
        wgIntersectionDotSize = 15
        wgCmap = 'gist_rainbow'
        wgColorbarOrientation = 'vertical'
        wgCmap_case = 'Estd'
        match wgCmap_case:
            case 'Eavg':
                wgNegCmap_values = wgNeg_EavgList
                wgPosCmap_values = wgPos_EavgList
                wgNorm = plt.Normalize(7.5e-3,15e-3)
                wgColorbarLabel = 'E Field Centroid Along x-axis [m]'
            case 'Estd':
                wgNegCmap_values = wgNeg_EstdList
                wgPosCmap_values = wgPos_EstdList
                wgNorm = plt.Normalize(.001, .01)
                wgColorbarLabel = 'E Field Standard Deviation Along x-axis [m]'
            case 'Emax': #bad measure to go off of
                wgNegCmap_values = wgNeg_EmaxList
                wgPosCmap_values = wgPos_EmaxList
                wgNorm = plt.Normalize(0.03,0.12)
                wgColorbarLabel = 'E Field Maximum Value [m]'
            case _:
                Exception("Improper Waveguide Cmap Case")
        #region --- Waveguide Intersection Plotting Backend ---
        negScatter = axs['e'].scatter(wgNeg_k0list, wgNeg_evalList.real/wp0, s=wgIntersectionDotSize,
                        c=wgNegCmap_values, norm=wgNorm, cmap=wgCmap)
        posScatter = axs['e'].scatter(wgPos_k0list, wgPos_evalList.real/wp0, s=wgIntersectionDotSize,
                        c=wgPosCmap_values, norm=wgNorm, cmap=wgCmap)
        wgCbar = fig.colorbar(posScatter, ax=axs['e'], orientation=wgColorbarOrientation)
        wgCbar.set_label(wgColorbarLabel)
        #endregion

if plotTextInfo:
    text = rf"""
    $\theta$ = {thetadegs}$^\circ$
    $f_p$ = {fp0/1e9} GHz
    $B_0$ = 87 mT
    """
    axs['t'].text(0, 1, text, va='top', ha='left', fontsize=12)
    axs['t'].axis('off')

plt.show()