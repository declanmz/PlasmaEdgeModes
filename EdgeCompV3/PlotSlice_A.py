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
from math import radians

from JSONHelpers import TypeEncoder, as_complex
from EvecSparseSolve import EvecSparseSolve
#endregion

#region ----- Define Constants -----
e = 1.6e-19 #C - charge on electron
m_e = 9.1094e-31 #kg -  mass of electron
mu_0 = 4e-7*np.pi #H/m - permeability of free space
ep_0 = 8.854e-12 #F/m - permitivity of free space
c = 3e8 #m/s - speed of light
#endregion

# ----- Choose File -----
baseDirectory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/EdgeCompV3/Setups'
setupFolder = 'Qin2023Linear'
kzoffset_n = 0.5
kyoffset_n = 0
thetadegs = 90

kmag_close_n = 2
w_close_n = 3.27

#region ----- Load JSON Data -----
file = f'{baseDirectory}/{setupFolder}/{kzoffset_n}kzn_{kyoffset_n}kyn_{thetadegs}deg_SliceSolve.json'
with open(file, 'r') as f:
    jsondata = json.load(f, object_hook=as_complex)
# unpacking:
fr = jsondata['fr']
wr = 2*np.pi*fr
N = jsondata['N']
L_n = jsondata['L_n']
wc_n = jsondata['wc_n']
deltax_n = jsondata['deltax_n']
xlist_n = np.array(jsondata['xlist_n'])
wplist_n = np.array(jsondata['wplist_n'])
wp1_n = wplist_n.max()
wp2_n = wplist_n.min()
eplist = jsondata['eplist']
kmin_n = jsondata['kmin_n']
kmax_n = jsondata['kmax_n']
thetadegs = jsondata['thetadegs']
Nk = jsondata['Nk']
kzoffset_n = jsondata['kzoffset_n']
kyoffset_n = jsondata['kyoffset_n']
klist_n = np.asarray(jsondata['klist_n'])

evals_n_list = list(map(np.array, jsondata['evals_n_list']))
avgs_n_list = list(map(np.array, jsondata['avgs_n_list']))
absAvgs_n_list = list(map(np.array, jsondata['absAvgs_n_list']))
stds_n_list = list(map(np.array, jsondata['stds_n_list']))
absStds_n_list = list(map(np.array, jsondata['absStds_n_list']))

avgFilterBounds = [-100, 100]
absAvgFilterBounds = [0, 100]
stdFilterBounds = [0, L_n]
absStdFilterBounds = [0, L_n]

#Do filtering
for i in range(Nk):
        avgFilteredIndices = np.where((avgs_n_list[i] >= avgFilterBounds[0]) & (avgs_n_list[i] <= avgFilterBounds[1]))
        absAvgFilteredIndices = np.where((absAvgs_n_list[i] >= absAvgFilterBounds[0]) & (absAvgs_n_list[i] <= absAvgFilterBounds[1]))
        stdFilteredIndices = np.where((stds_n_list[i] >= stdFilterBounds[0]) & (stds_n_list[i] <= stdFilterBounds[1]))
        absStdFilteredIndices = np.where((absStds_n_list[i] >= absStdFilterBounds[0]) & (absStds_n_list[i] <= absStdFilterBounds[1]))

        filteredIndices = np.intersect1d(np.intersect1d(avgFilteredIndices, absAvgFilteredIndices), np.intersect1d(stdFilteredIndices, absStdFilteredIndices))

        evals_n_list[i] = evals_n_list[i][filteredIndices]
        avgs_n_list[i] = avgs_n_list[i][filteredIndices]
        absAvgs_n_list[i] = absAvgs_n_list[i][filteredIndices]
        stds_n_list[i] = stds_n_list[i][filteredIndices]
        absStds_n_list[i] = absStds_n_list[i][filteredIndices]




# try: #NEED TO FIX
#     wgNeg_klist = np.asarray(jsondata['wgNeg_klist'])
#     wgNeg_distanceList = np.asarray(jsondata['wgNeg_distanceList'])
#     wgNeg_evalList_n = np.asarray(jsondata['wgNeg_evalList_n'])
#     wgNeg_EavgList_n = np.asarray(jsondata['wgNeg_EavgList_n'])
#     wgNeg_EstdList_n = np.asarray(jsondata['wgNeg_EstdList_n'])
#     wgNeg_EmaxList_n = np.asarray(jsondata['wgNeg_EmaxList_n'])

#     wgPos_klist = np.asarray(jsondata['wgPos_klist'])
#     wgPos_distanceList = np.asarray(jsondata['wgPos_distanceList'])
#     wgPos_evalList_n = np.asarray(jsondata['wgPos_evalList_n'])
#     wgPos_EavgList_n = np.asarray(jsondata['wgPos_EavgList_n'])
#     wgPos_EstdList_n = np.asarray(jsondata['wgPos_EstdList_n'])
#     wgPos_EmaxList_n = np.asarray(jsondata['wgPos_EmaxList_n'])
# except:
#     print("No waveguide data")
#endregion

#region ----- Bulk Modes Calculation -----
def BuildBulkMatrix(wp_n, wc_n, kn_x, kn_y, kn_z):
    return np.array([[0,-1j*wc_n,0,-1j*wp_n,0,0,0,0,0],
                  [1j*wc_n,0,0,0,-1j*wp_n,0,0,0,0],
                  [0,0,0,0,0,-1j*wp_n,0,0,0],
                  [1j*wp_n,0,0,0,0,0,0,kn_z,-kn_y],
                  [0,1j*wp_n,0,0,0,0,-kn_z,0,kn_x],
                  [0,0,1j*wp_n,0,0,0,kn_y,-kn_x,0],
                  [0,0,0,0,-kn_z,kn_y,0,0,0],
                  [0,0,0,kn_z,0,-kn_x,0,0,0],
                  [0,0,0,-kn_y,kn_x,0,0,0,0]])

def BulkModesSlice(wp_n, wc_n, kmin_n, kmax_n, kzoffset_n, kyoffset_n):
    dots = 1000
    kdotslist_n = np.linspace(kmin_n, kmax_n, dots)
    kydotslist_n = kdotslist_n * np.sin(radians(thetadegs)) + kyoffset_n
    kzdotslist_n = kdotslist_n * np.cos(radians(thetadegs)) + kzoffset_n

    bulkDispersion_n_list = np.empty((4,dots))
    for i in range(dots):
        w_n_list, evecs_n = np.linalg.eig(
            BuildBulkMatrix(wp_n, wc_n, 0, kydotslist_n[i], kzdotslist_n[i]))
        w_n_list = np.sort(w_n_list)
        for j in range(4):
            bulkDispersion_n_list[j][i] = w_n_list[j+5]
    
    return kdotslist_n, bulkDispersion_n_list


# lightline_n = wp0/wr * np.asarray([np.sqrt(ky0line[i]**2 + kz0line[i]**2) for i in range(dots)])
#endregion

#region ----- Load Waveguide Dispersion -----
# dwg = 6.25e-3

# dispersionload = np.loadtxt(f"{baseDirectory}/WaveguideDispersion/eps_4.txt")
# k_waveguide = dispersionload[:,0]/180 /dwg * np.pi
# w_waveguide_n = dispersionload[:,1] * 2 * np.pi * 1e9 / wr
#endregion

# ----- Choose Preset for Layout -----
plotPreset = 3

#region --- Preset Variables ---
plotDispersion = False
plotBulkModes = False
plotLightLine = False
plotWgLine = False
plotWgIntersection = False

plotEigenvectors = False
plotEigenvectorX = False
exlet = 'x'
plotEigenvectorY = False
eylet = 'y'
plotEigenvectorZ = False
ezlet = 'z'
plotEigenvectorPar = False
eparlet = 'par'
plotEigenvectorPerp = False
eperplet = 'perp'

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

if plotPreset == 2: #Dispersion Plot with Eigenvectors and Text Info
    fig, axs = plt.subplot_mosaic([['e', 'e', 'e', 'x', 'x', 't'],
                                   ['e', 'e', 'e', 'y', 'y', 't'],
                                   ['e', 'e', 'e', 'z', 'z', 't']], figsize=(20, 10))

    plotDispersion = True
    plotBulkModes = True
    plotLightLine = True
    plotWgLine = True
    plotWgIntersection = True

    plotEigenvectors = True
    plotEigenvectorX = True
    plotEigenvectorY = True
    plotEigenvectorZ = True

    plotTextInfo = True

if plotPreset == 3: #Dispersion Plot with Eigenvectors and Text Info
    fig, axs = plt.subplot_mosaic([['e', 'e', 'e', 'x', 'x', 't'],
                                   ['e', 'e', 'e', 'par', 'par', 't'],
                                   ['e', 'e', 'e', 'perp', 'perp', 't']], figsize=(20, 10))

    plotDispersion = True
    plotBulkModes = True
    plotLightLine = True
    plotWgLine = True
    plotWgIntersection = True

    plotEigenvectors = True
    plotEigenvectorX = True
    plotEigenvectorPar = True
    plotEigenvectorPerp = True

    plotTextInfo = True

def dispersionPlotter(kmag_show_n, eval_n_show, genColorbars): #Properties of the Dispersion Plot + Plotting Itself
    # --- Dispersion Plotting Properties ---
    axs['e'].set_title('Eigenvalues')
    axs['e'].set_xlabel(r'$ck/\omega_r$', size=12)
    axs['e'].set_ylabel(r'$\omega/\omega_r$', size=12) #FOR WHEN fr = 1GHz
    axs['e'].grid(color='gray', linestyle='dashed', alpha=0.2)

    overwrite_xlim = [None, None] #Values in terms of 1/m
    overwrite_ylim = [0, 2.5] #Values in terms of wr

    dotsize = 10

    dispCmap = 'coolwarm'
    dispColorbarOrientation = 'vertical'
    dispCmap_case = 'avg'
    match dispCmap_case:
        case 'avg':
            dispCmap_values = [arr * c / wr for arr in avgs_n_list]
            dispCmap_norm = plt.Normalize(-L_n * c / wr, L_n * c / wr)
            dispColorbarLabel = 'Evec Avg [m]'
        case 'absAvg':
            dispCmap_values = [arr * c / wr for arr in absAvgs_n_list]
            dispCmap_norm = plt.Normalize(0, L_n * c / wr)
            dispColorbarLabel = 'Evec Abs Avg [m]'
        case 'std':
            dispCmap_values = [arr * c / wr for arr in stds_n_list]
            dispCmap_norm = plt.Normalize(0, 0.6*L_n * c / wr)
            dispColorbarLabel = 'Evec Std [m]'
        case 'absStd':
            dispCmap_values = [arr * c / wr for arr in absStds_n_list]
            dispCmap_norm = plt.Normalize(0, 0.3*L_n * c / wr)
            dispColorbarLabel = 'Evec Abs Std [m]'
        case _:
            Exception("Improper Dispersion Cmap Case")

    #region --- Dispersion Plotting Backend ---
    for n in range(len(klist_n)):
        dispScatter = axs['e'].scatter([klist_n[n] for i in evals_n_list[n]], evals_n_list[n].real, s=dotsize, 
                        c=dispCmap_values[n], norm=dispCmap_norm, cmap=dispCmap)
    
    if overwrite_xlim[0] == None:
        overwrite_xlim[0] = klist_n[0]
    if overwrite_xlim[1] == None:
        overwrite_xlim[1] = klist_n[-1]
    axs['e'].set_xlim(overwrite_xlim)

    if overwrite_ylim[0] == None:
        overwrite_ylim[0] = 0
    if overwrite_ylim[1] != None:
        axs['e'].set_ylim(overwrite_ylim)


    if genColorbars:
        dispCbar = fig.colorbar(dispScatter, ax=axs['e'], orientation=dispColorbarOrientation)
        dispCbar.set_label(dispColorbarLabel)
    #endregion

    if plotBulkModes:
        kdotslist_wp1_n, bulkDispersion_wp1_n_list = BulkModesSlice(wp1_n, wc_n, kmin_n, kmax_n, kzoffset_n, kyoffset_n)
        kdotslist_wp2_n, bulkDispersion_wp2_n_list = BulkModesSlice(wp2_n, wc_n, kmin_n, kmax_n, kzoffset_n, kyoffset_n)

        for i in range(4):
            axs['e'].plot(kdotslist_wp1_n, bulkDispersion_wp1_n_list[i], color='green', alpha=0.5)
            axs['e'].plot(kdotslist_wp2_n, bulkDispersion_wp2_n_list[i], color='yellow', alpha=0.5)

    # if plotLightLine:
    #     axs['e'].plot(bulk_kline, lightline_n, color='yellow')

    # if plotWgLine:
    #     wgLineColor = 'green'
    #     axs['e'].plot(k_waveguide, w_waveguide_n, color=wgLineColor)
    #     axs['e'].plot(-k_waveguide, w_waveguide_n, color=wgLineColor)

    # if plotWgIntersection:
    #     wgIntersectionDotSize = 15
    #     wgCmap = 'gist_rainbow'
    #     wgColorbarOrientation = 'vertical'
    #     wgCmap_case = 'Estd'
    #     match wgCmap_case:
    #         case 'Eavg':
    #             wgNegCmap_values = c*wgNeg_EavgList_n/wr
    #             wgPosCmap_values = c*wgPos_EavgList_n/wr
    #             wgNorm = plt.Normalize(7.5e-3,15e-3)
    #             wgColorbarLabel = 'E Field Centroid Along x-axis [m]'
    #         case 'Estd':
    #             wgNegCmap_values = c*wgNeg_EstdList_n/wr
    #             wgPosCmap_values = c*wgPos_EstdList_n/wr
    #             wgNorm = plt.Normalize(.001, .01)
    #             wgColorbarLabel = 'E Field Standard Deviation Along x-axis [m]'
    #         case 'Emax': #bad measure to go off of
    #             wgNegCmap_values = wgNeg_EmaxList_n
    #             wgPosCmap_values = wgPos_EmaxList_n
    #             wgNorm = plt.Normalize(0.03,0.12)
    #             wgColorbarLabel = 'E Field Maximum Value [m]'
    #         case _:
    #             Exception("Improper Waveguide Cmap Case")
    #     #region --- Waveguide Intersection Plotting Backend ---
    #     negScatter = axs['e'].scatter(wgNeg_klist, wgNeg_evalList_n.real, s=wgIntersectionDotSize,
    #                     c=wgNegCmap_values, norm=wgNorm, cmap=wgCmap)
    #     posScatter = axs['e'].scatter(wgPos_klist, wgPos_evalList_n.real, s=wgIntersectionDotSize,
    #                     c=wgPosCmap_values, norm=wgNorm, cmap=wgCmap)
    #     if genColorbars:
    #         wgCbar = fig.colorbar(posScatter, ax=axs['e'], orientation=wgColorbarOrientation)
    #         wgCbar.set_label(wgColorbarLabel)
    #     #endregion

    if plotEigenvectors:
        axs['e'].scatter(kmag_show_n, eval_n_show, marker='x', s=100, color='black', alpha=0.4)


def evecPlotter(file, kmag_close_n, w_close_n):
    evecdict = EvecSparseSolve(file, kmag_close_n, w_close_n)
    L = (c * L_n / wr)
    xlist = (c * xlist_n / wr)
    xticklist = np.linspace(-L, L, 7)
    deltax = (c * deltax_n / wr)
    propTheta = evecdict['propTheta']

    evec_phase = np.angle(np.asarray(evecdict['Ex']).max())
    evec_mult = np.exp(-1j * evec_phase)

    Ex = np.asarray(evecdict['Ex'])*evec_mult
    Ey = np.asarray(evecdict['Ey'])*evec_mult
    Ez = np.asarray(evecdict['Ez'])*evec_mult
    Epar = Ez * np.cos(propTheta) + Ey * np.sin(propTheta)
    Eperp = Ez * np.sin(propTheta) - Ey * np.cos(propTheta)

    Bx_n = np.asarray(evecdict['Bx_n'])*evec_mult
    By_n = np.asarray(evecdict['By_n'])*evec_mult
    Bz_n = np.asarray(evecdict['Bz_n'])*evec_mult
    Bpar_n = Bz_n * np.cos(propTheta) + By_n * np.sin(propTheta)
    Bperp_n = Bz_n * np.sin(propTheta) - By_n * np.cos(propTheta)

    ux = np.asarray(evecdict['ux'])*evec_mult
    uy = np.asarray(evecdict['uy'])*evec_mult
    uz = np.asarray(evecdict['uz'])*evec_mult
    upar = uz * np.cos(propTheta) + uy * np.sin(propTheta)
    uperp = uz * np.sin(propTheta) - uy * np.cos(propTheta)
    
    max_list = []
    for vec in [Ex, Ey, Ez, Epar, Eperp, Bx_n, By_n, Bz_n, Bpar_n, Bperp_n, ux, uy, uz, upar, uperp]:
        max_list.append((np.abs(vec).real).max())
        max_list.append((np.abs(vec).imag).max())

    evec_norm = np.max(max_list)

    common_evecList = []

    if plotEigenvectorX:
        common_evecList.append(exlet)
        axs[exlet].set_title(r'$E_x, B_x, u_x$')

        axs[exlet].plot(xlist, Ex.imag / evec_norm, 'r--')
        axs[exlet].plot(xlist, Ex.real / evec_norm, 'r-')

        axs[exlet].plot(xlist, Bx_n.real / evec_norm, 'b-')
        axs[exlet].plot(xlist, Bx_n.imag / evec_norm, 'b--')

        axs[exlet].plot(xlist, ux.real / evec_norm, 'g-')
        axs[exlet].plot(xlist, ux.imag / evec_norm, 'g--')

    if plotEigenvectorY:
        common_evecList.append(eylet)
        axs[eylet].set_title(r'$E_y, B_y, u_y$')

        axs[eylet].plot(xlist, Ey.real / evec_norm, 'r-')
        axs[eylet].plot(xlist, Ey.imag / evec_norm, 'r--')

        axs[eylet].plot(xlist, By_n.real / evec_norm, 'b-')
        axs[eylet].plot(xlist, By_n.imag / evec_norm, 'b--')

        axs[eylet].plot(xlist, uy.real / evec_norm, 'g-')
        axs[eylet].plot(xlist, uy.imag / evec_norm, 'g--')

    if plotEigenvectorZ:
        common_evecList.append(ezlet)
        axs[ezlet].set_title(r'$E_z, B_z, u_z$')
        axs[ezlet].set_xlabel('mm')

        axs[ezlet].plot(xlist, Ez.real / evec_norm, 'r-')
        axs[ezlet].plot(xlist, Ez.imag / evec_norm, 'r--')

        axs[ezlet].plot(xlist, Bz_n.real / evec_norm, 'b-')
        axs[ezlet].plot(xlist, Bz_n.imag / evec_norm, 'b--')

        axs[ezlet].plot(xlist, uz.real / evec_norm, 'g-')
        axs[ezlet].plot(xlist, uz.imag / evec_norm, 'g--')

    if plotEigenvectorPar:
        common_evecList.append(eparlet)
        axs[eparlet].set_title(r'$E_\parallel, B_\parallel, u_\parallel$')
        #NEED TO DO
        axs[eparlet].plot(xlist, Epar.real / evec_norm, 'r-')
        axs[eparlet].plot(xlist, Epar.imag / evec_norm, 'r--')

        axs[eparlet].plot(xlist, Bpar_n.real / evec_norm, 'b-')
        axs[eparlet].plot(xlist, Bpar_n.imag / evec_norm, 'b--')

        axs[eparlet].plot(xlist, upar.real / evec_norm, 'g-')
        axs[eparlet].plot(xlist, upar.imag / evec_norm, 'g--')
    
    if plotEigenvectorPerp:
        common_evecList.append(eperplet)
        axs[eperplet].set_title(r'$E_\perp, B_\perp, u_\perp$')
        axs[eperplet].set_xlabel('mm')

        #NEED TO DO
        axs[eperplet].plot(xlist, Eperp.real / evec_norm, 'r-')
        axs[eperplet].plot(xlist, Eperp.imag / evec_norm, 'r--')

        axs[eperplet].plot(xlist, Bperp_n.real / evec_norm, 'b-')
        axs[eperplet].plot(xlist, Bperp_n.imag / evec_norm, 'b--')

        axs[eperplet].plot(xlist, uperp.real / evec_norm, 'g-')
        axs[eperplet].plot(xlist, uperp.imag / evec_norm, 'g--')

    twinaxes = {}
    for elet in common_evecList:
        axs[elet].set_ylim([-1.2,1.2])
        axs[elet].set_xticks(xticklist)
        axs[elet].set_xticklabels([round(xt,3) for xt in xticklist*1e3])

        twinaxes[elet] = axs[elet].twinx()
        twinaxes[elet].set_ylim([wp2_n - 0.2*(wp1_n-wp2_n), wp1_n + 0.2*(wp1_n-wp2_n)])
        twinaxes[elet].set_ylabel(r'$\omega_p/\omega_r$')
        twinaxes[elet].plot(xlist, wplist_n, color='cyan', alpha=0.2)

        for n in range(N):
            if eplist[n] > 1:
                axs[elet].add_patch(patches.Rectangle((xlist[n], -1.2), deltax, 2.4, color='blue', alpha=0.2, ec=None))
            if eplist[n] < 1:
                axs[elet].add_patch(patches.Rectangle((xlist[n], -1.2), deltax, 2.4, color='silver', alpha=1, ec=None))

    return evecdict['kmag_n'], evecdict['eval_returned_n']

def textInfoPlotter(kmag_show_n, eval_n_show):
    text = rf"""
    $f_r$ = {fr/1e9} GHz
    $\omega_1/\omega_r$ = {wp1_n}
    $\omega_2/\omega_r$ = {wp2_n}
    $\omega_c/\omega_r$ = {wc_n}
    $\theta$ = {thetadegs}$^\circ$
    shown $ck/\omega_r$ = {kmag_show_n:.3f}
    shown $\omega/\omega_r$ = {(eval_n_show):.3f}
    """
    axs['t'].text(0, 1, text, va='top', ha='left', fontsize=12)
    axs['t'].axis('off')

kmag_show_n = None
eval_n_show = None

def on_click(event):
    # Check if the click is within the axes
    if event.inaxes is not None:
        # Get the x and y data of the click event
        x_click = event.xdata
        y_click = event.ydata

        k_n_click_id = np.argmin(np.abs(klist_n - x_click))
        k_n_click = klist_n[k_n_click_id]
        evals_n_click = np.asarray(evals_n_list[k_n_click_id])
        eval_n_click = evals_n_click[np.argmin(np.abs(evals_n_click - y_click))]

        axs['x'].cla()
        axs['par'].cla()
        axs['perp'].cla()
        axs['e'].cla()
        axs['t'].cla()
        kmag_show, eval_n_show = evecPlotter(file, k_n_click, eval_n_click)
        dispersionPlotter(kmag_show, eval_n_show, False)
        textInfoPlotter(kmag_show, eval_n_show)
        print(f"Clicked at x={x_click}, y={y_click}")
        fig.canvas.draw()


kmag_show_n, eval_n_show = evecPlotter(file, kmag_close_n, w_close_n)
fig.canvas.mpl_connect('button_press_event', on_click)

dispersionPlotter(kmag_show_n, eval_n_show, True)
textInfoPlotter(kmag_show_n, eval_n_show)


plt.tight_layout(pad=2.0)
plt.show()