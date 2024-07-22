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
filterName = 'Unfiltered'
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
deltax = jsondata['deltax']

klist_n = np.asarray(jsondata['klist_n'])
klist = wr*klist_n/c
evals_list_n = list(map(np.array, jsondata['evals_list_n']))
Eavg_list_n = list(map(np.array, jsondata['Eavg_list_n']))
Eavg_list = [c*val/wr for val in Eavg_list_n]
Estd_list_n = list(map(np.array, jsondata['Estd_list_n']))
Estd_list = [c*val/wr for val in Estd_list_n]
Emax_list_n = list(map(np.array, jsondata['Emax_list_n']))

try: #NEED TO FIX
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

# ----- Choose Preset for Layout -----
plotPreset = 1

#region --- Preset Variables ---
plotSetup = False

plotTextInfo = False
#endregion

if plotPreset == 1: #Dispersion Plot with Text Info
    fig, axs = plt.subplot_mosaic([['s', 's', 's', 's', 't']], figsize=(12, 7))

    plotSetup = True

    plotTextInfo = True

if plotSetup:
    axs['s'].set_title('Physical Setup')
    axs['s'].plot(xlist, np.divide(wplist_n, wp0/wr))
    for n in range(N):
        if eplist[n] > 1:
            axs['s'].add_patch(patches.Rectangle((xlist[n] - deltax/2, -0.1), deltax, 1.2, color='blue', alpha=0.2, ec=None))
        if eplist[n] < 1:
            axs['s'].add_patch(patches.Rectangle((xlist[n] - deltax/2, -0.1), deltax, 1.2, color='silver', alpha=1, ec=None))
    axs['s'].set_ylim([-0.1,1.1])
    xticklist = np.linspace(-L, L, 7)
    axs['s'].set_xticks(xticklist, [round(xt,3) for xt in xticklist*1e3])
    axs['s'].set_xlabel(r'mm')
    axs['s'].set_ylabel(r'$\omega/\omega_p$')


if plotTextInfo:
    text = rf"""
    $\theta$ = {thetadegs}$^\circ$
    $f_p$ = {fp0/1e9} GHz
    $B_0$ = 87 mT
    """
    axs['t'].text(0, 1, text, va='top', ha='left', fontsize=12)
    axs['t'].axis('off')

plt.show()