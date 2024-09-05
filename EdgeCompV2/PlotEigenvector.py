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

kmag_close = 300
w_close_n = 3

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
deltax = jsondata['deltax']

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
plotEigenvectors = False
plotEigenvectorX = False
exlet = 'x'
plotEigenvectorY = False
eylet = 'y'
plotEigenvectorZ = False
ezlet = 'z'

plotTextInfo = False
#endregion

if plotPreset == 1: #Dispersion Plot with Text Info
    fig, axs = plt.subplot_mosaic([['x'], ['y'], ['z']], figsize=(12, 7))

    plotEigenvectors = True
    plotEigenvectorX = True
    plotEigenvectorY = True
    plotEigenvectorZ = True

    plotTextInfo = False


if plotEigenvectors:
    evecdict = EvecSparseEigensolve_toJSON(file, '', kmag_close, w_close_n, False)
    xticklist = np.linspace(-L, L, 7)
    evec_norm = np.abs(np.asarray(evecdict['Ex'])).max()
    evec_phase = np.angle(np.asarray(evecdict['Ex']).max())
    evec_mult = -1*np.exp(-1j * evec_phase)

    common_evecList = []

    if plotEigenvectorX:
        common_evecList.append(exlet)
        axs[exlet].set_title('X Direction')

        axs[exlet].plot(xlist, (np.asarray(evecdict['Ex']) * evec_mult).imag / evec_norm, 'r--')
        axs[exlet].plot(xlist, (np.asarray(evecdict['Ex']) * evec_mult).real / evec_norm, 'r-')

        axs[exlet].plot(xlist, (np.asarray(evecdict['Bx_n']) * evec_mult).real / evec_norm, 'b-')
        axs[exlet].plot(xlist, (np.asarray(evecdict['Bx_n']) * evec_mult).imag / evec_norm, 'b--')

        axs[exlet].plot(xlist, (np.asarray(evecdict['ux']) * evec_mult).real / evec_norm, 'g-')
        axs[exlet].plot(xlist, (np.asarray(evecdict['ux']) * evec_mult).imag / evec_norm, 'g--')

    if plotEigenvectorY:
        common_evecList.append(eylet)
        axs[eylet].set_title('Y Direction')

        axs[eylet].plot(xlist, (np.asarray(evecdict['Ey']) * evec_mult).real / evec_norm, 'r-')
        axs[eylet].plot(xlist, (np.asarray(evecdict['Ey']) * evec_mult).imag / evec_norm, 'r--')

        axs[eylet].plot(xlist, (np.asarray(evecdict['By_n']) * evec_mult).real / evec_norm, 'b-')
        axs[eylet].plot(xlist, (np.asarray(evecdict['By_n']) * evec_mult).imag / evec_norm, 'b--')

        axs[eylet].plot(xlist, (np.asarray(evecdict['uy']) * evec_mult).real / evec_norm, 'g-')
        axs[eylet].plot(xlist, (np.asarray(evecdict['uy']) * evec_mult).imag / evec_norm, 'g--')

    if plotEigenvectorZ:
        common_evecList.append(ezlet)
        axs[ezlet].set_title('Z Direction')

        axs[ezlet].plot(xlist, (np.asarray(evecdict['Ez']) * evec_mult).real / evec_norm, 'r-')
        axs[ezlet].plot(xlist, (np.asarray(evecdict['Ez']) * evec_mult).imag / evec_norm, 'r--')

        axs[ezlet].plot(xlist, (np.asarray(evecdict['Bz_n']) * evec_mult).real / evec_norm, 'b-')
        axs[ezlet].plot(xlist, (np.asarray(evecdict['Bz_n']) * evec_mult).imag / evec_norm, 'b--')

        axs[ezlet].plot(xlist, (np.asarray(evecdict['uz']) * evec_mult).real / evec_norm, 'g-')
        axs[ezlet].plot(xlist, (np.asarray(evecdict['uz']) * evec_mult).imag / evec_norm, 'g--')

    for elet in common_evecList:
        axs[exlet].set_ylim([-1.2,1.2])
        axs[exlet].set_xticks(xticklist)
        axs[exlet].set_xticklabels([round(xt,3) for xt in xticklist*1e3])

        axs[elet].plot(xlist, 2*np.divide(wplist, wp0) - 1, color='cyan', alpha=0.2)

        for n in range(N):
            if eplist[n] > 1:
                axs[elet].add_patch(patches.Rectangle((xlist[n], -1.2), deltax, 2.4, color='blue', alpha=0.2, ec=None))
            if eplist[n] < 1:
                axs[elet].add_patch(patches.Rectangle((xlist[n], -1.2), deltax, 2.4, color='silver', alpha=1, ec=None))

if plotTextInfo:
    text = rf"""
    $\theta$ = {thetadegs}$^\circ$
    $f_p$ = {fp0/1e9} GHz
    $B_0$ = 87 mT
    """
    axs['t'].text(0, 1, text, va='top', ha='left', fontsize=12)
    axs['t'].axis('off')


plt.tight_layout(pad=2.0)
plt.show()