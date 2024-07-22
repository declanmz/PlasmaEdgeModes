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
# from EvecSparseEigensolve import EvecSparseEigensolve_toJSON
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
setupFolder = 'Testing'
filterName = 'FilterA'
thetadegs = 39
sizeScaling = 1

#region ----- Load JSON Data -----
file = f'{baseDirectory}/{setupFolder}/{thetadegs}deg_{filterName}.json'
with open(file, 'r') as f:
    jsondata = json.load(f, object_hook=as_complex)
# unpacking:
N = jsondata['N']
fr = jsondata['fr']
wr = jsondata['wr']
L = jsondata['L']
L_n = jsondata['L_n']
fp0 = jsondata['fp0']
wp0 = jsondata['wp0']
thetadegs = jsondata['thetadegs']
theta = jsondata['theta']
Nk = jsondata['Nk']
wmin = jsondata['wmin']
wmin_n = jsondata['wmin_n']
wmax = jsondata['wmax']
wmax_n = jsondata['wmax_n']
kmin = jsondata['kmin']
kmin_n = jsondata['kmin_n']
kmax = jsondata['kmax']
kmax_n = jsondata['kmax_n']
B0 = jsondata['B0']
fc = jsondata['fc']
wc_n = jsondata['wc_n']
kzoffset = jsondata['kzoffset']
kzoffset_n = jsondata['kzoffset_n']

xlist = jsondata['xlist']
xlist_n = jsondata['xlist_n']
wplist = jsondata['wplist']
wplist_n = jsondata['wplist_n']
eplist = jsondata['eplist']

klist_n = np.asarray(jsondata['klist_n'])
klist = wr*klist_n/c
evals_list_n = list(map(np.array, jsondata['evals_list_n']))
Eavg_list_n = list(map(np.array, jsondata['Eavg_list_n']))
Eavg_list = [c*val/wr for val in Eavg_list_n]
Estd_list_n = list(map(np.array, jsondata['Estd_list_n']))
Estd_list = [c*val/wr for val in Estd_list_n]
Emax_list_n = list(map(np.array, jsondata['Emax_list_n']))

try: #NEED TO FIX
    wgNeg_klist = np.asarray(jsondata['wgNeg_klist'])
    wgNeg_distanceList = np.asarray(jsondata['wgNeg_distanceList'])
    wgNeg_evalList_n = np.asarray(jsondata['wgNeg_evalList_n'])
    wgNeg_EavgList_n = np.asarray(jsondata['wgNeg_EavgList_n'])
    wgNeg_EstdList_n = np.asarray(jsondata['wgNeg_EstdList_n'])
    wgNeg_EmaxList_n = np.asarray(jsondata['wgNeg_EmaxList_n'])

    wgPos_klist = np.asarray(jsondata['wgPos_klist'])
    wgPos_distanceList = np.asarray(jsondata['wgPos_distanceList'])
    wgPos_evalList_n = np.asarray(jsondata['wgPos_evalList_n'])
    wgPos_EavgList_n = np.asarray(jsondata['wgPos_EavgList_n'])
    wgPos_EstdList_n = np.asarray(jsondata['wgPos_EstdList_n'])
    wgPos_EmaxList_n = np.asarray(jsondata['wgPos_EmaxList_n'])
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

k0line = np.linspace(c*kmin/wp0, c*kmax/wp0,dots)
ky0line = k0line * np.sin(theta)
kz0line = k0line * np.cos(theta) + c*kzoffset/wp0
bulk_kline = wp0 * k0line/c
bulkLines_n = np.empty((5,dots))
sig = fc / fp0

for n in range(dots):
    omns_n = wp0/wr * np.sort(MagPlasmaEigenmodes(sig, ky0line[n], 0, kz0line[n])[0])
    for i in range(5):
        bulkLines_n[i][n] = omns_n[i+4]

lightline_n = wp0/wr * np.asarray([np.sqrt(ky0line[i]**2 + kz0line[i]**2) for i in range(dots)])
#endregion

#region ----- Load Waveguide Dispersion -----
dwg = 6.25e-3

dispersionload = np.loadtxt(f"{baseDirectory}/WaveguideDispersion/eps_4.txt")
k_waveguide = dispersionload[:,0]/180 /dwg * np.pi
w_waveguide_n = dispersionload[:,1] * 2 * np.pi * 1e9 / wr
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
    axs['e'].set_xlabel(r'k [$m^{-1}$]', size=12)
    axs['e'].set_ylabel(r'$f$ [GHz]', size=12) #FOR WHEN fr = 1GHz
    axs['e'].grid(color='gray', linestyle='dashed', alpha=0.2)

    overwrite_xlim = [None, None] #Values in terms of 1/m
    overwrite_ylim = [None, None] #Values in terms of wr

    dotsize = 10

    dispCmap = 'coolwarm'
    dispColorbarOrientation = 'vertical'
    dispCmap_case = 'Eavg'
    match dispCmap_case:
        case 'Eavg':
            dispCmap_values = Eavg_list
            dispCmap_norm = plt.Normalize(-2.5e-3 * sizeScaling, 7.5e-3*sizeScaling)
            dispColorbarLabel = 'E Field Centroid along x-axis [m]'
        case _:
            Exception("Improper Dispersion Cmap Case")

    #region --- Dispersion Plotting Backend ---
    for n in range(len(klist)):
        dispScatter = axs['e'].scatter([klist[n] for i in evals_list_n[n]], evals_list_n[n].real, s=dotsize, 
                        c=dispCmap_values[n], norm=dispCmap_norm, cmap=dispCmap)
    
    if overwrite_xlim[0] == None:
        overwrite_xlim[0] = klist[0]
    if overwrite_xlim[1] == None:
        overwrite_xlim[1] = klist[-1]
    if overwrite_ylim[0] == None:
        overwrite_ylim[0] = wmin/wr
    if overwrite_ylim[1] == None:
        overwrite_ylim[1] = wmax/wr
    axs['e'].set_xlim(overwrite_xlim)
    axs['e'].set_ylim(overwrite_ylim)

    dispCbar = fig.colorbar(dispScatter, ax=axs['e'], orientation=dispColorbarOrientation)
    dispCbar.set_label(dispColorbarLabel)
    #endregion

    if plotBulkModes:
        for i in range(len(bulkLines_n)):
            axs['e'].plot(bulk_kline, bulkLines_n[i], color='black', alpha=0.5)

    if plotLightLine:
        axs['e'].plot(bulk_kline, lightline_n, color='yellow')

    if plotWgLine:
        wgLineColor = 'green'
        axs['e'].plot(k_waveguide, w_waveguide_n, color=wgLineColor)
        axs['e'].plot(-k_waveguide, w_waveguide_n, color=wgLineColor)

    if plotWgIntersection:
        wgIntersectionDotSize = 15
        wgCmap = 'gist_rainbow'
        wgColorbarOrientation = 'vertical'
        wgCmap_case = 'Estd'
        match wgCmap_case:
            case 'Eavg':
                wgNegCmap_values = c*wgNeg_EavgList_n/wr
                wgPosCmap_values = c*wgPos_EavgList_n/wr
                wgNorm = plt.Normalize(7.5e-3,15e-3)
                wgColorbarLabel = 'E Field Centroid Along x-axis [m]'
            case 'Estd':
                wgNegCmap_values = c*wgNeg_EstdList_n/wr
                wgPosCmap_values = c*wgPos_EstdList_n/wr
                wgNorm = plt.Normalize(.001, .01)
                wgColorbarLabel = 'E Field Standard Deviation Along x-axis [m]'
            case 'Emax': #bad measure to go off of
                wgNegCmap_values = wgNeg_EmaxList_n
                wgPosCmap_values = wgPos_EmaxList_n
                wgNorm = plt.Normalize(0.03,0.12)
                wgColorbarLabel = 'E Field Maximum Value [m]'
            case _:
                Exception("Improper Waveguide Cmap Case")
        #region --- Waveguide Intersection Plotting Backend ---
        negScatter = axs['e'].scatter(wgNeg_klist, wgNeg_evalList_n.real, s=wgIntersectionDotSize,
                        c=wgNegCmap_values, norm=wgNorm, cmap=wgCmap)
        posScatter = axs['e'].scatter(wgPos_klist, wgPos_evalList_n.real, s=wgIntersectionDotSize,
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