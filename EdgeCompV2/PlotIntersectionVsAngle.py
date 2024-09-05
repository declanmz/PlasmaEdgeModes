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
setupFolder = 'OGSetupNormalized'
filterName = 'FilterRight'

thetadegsList = np.arange(0,40,2)

#region ----- Grab data from files -----
neg_evals_BigList = []
neg_Eavg_BigList = []
neg_Estd_BigList = []
neg_Emax_BigList = []

pos_evals_BigList = []
pos_Eavg_BigList = []
pos_Estd_BigList = []
pos_Emax_BigList = []

for thetadegs in thetadegsList:
    filename =  f'{baseDirectory}/{setupFolder}/{thetadegs}deg_{filterName}.json'
    with open(filename, 'r') as f:
        jsondata = json.load(f, object_hook=as_complex)
    
    if jsondata['thetadegs'] != thetadegs:
        raise Exception('thetadegs mismatch')

    neg_evals_BigList.append(np.asarray(jsondata['wgNeg_evalList_n']).real)
    neg_Eavg_BigList.append(np.asarray(jsondata['wgNeg_EavgList_n']))
    neg_Estd_BigList.append(np.asarray(jsondata['wgNeg_EstdList_n']))
    neg_Emax_BigList.append(np.asarray(jsondata['wgNeg_EmaxList_n']))

    pos_evals_BigList.append(np.asarray(jsondata['wgPos_evalList_n']).real)
    pos_Eavg_BigList.append(np.asarray(jsondata['wgPos_EavgList_n']))
    pos_Estd_BigList.append(np.asarray(jsondata['wgPos_EstdList_n']))
    pos_Emax_BigList.append(np.asarray(jsondata['wgPos_EmaxList_n']))

wr = jsondata['wr']
#endregion

#region ----- Plotting -----
fig, ax = plt.subplots(1, 1, figsize=(10,5))
fig.suptitle('Intersection Plot', y=.95, size=20)

wgCmap = 'jet'
wgColorbarOrientation = 'vertical'
wgCmap_case = 'Eavg'
match wgCmap_case:
    case 'Eavg':
        wgNegCmap_values = [c*li/wr for li in neg_Eavg_BigList]
        wgPosCmap_values = [c*li/wr for li in pos_Eavg_BigList]
        wgNorm = plt.Normalize(-7.5e-3,7.5e-3)
        wgColorbarLabel = 'E Field Centroid Along x-axis [m]'
    case 'Estd':
        wgNegCmap_values = [c*li/wr for li in neg_Estd_BigList]
        wgPosCmap_values = [c*li/wr for li in pos_Estd_BigList]
        wgNorm = plt.Normalize(-.008, .008)
        wgColorbarLabel = 'E Field Standard Deviation Along x-axis [m]'
    case 'Emax': #bad measure to go off of
        wgNegCmap_values = neg_Emax_BigList
        wgPosCmap_values = pos_Emax_BigList
        wgNorm = plt.Normalize(0.03,0.12)
        wgColorbarLabel = 'E Field Maximum Value [m]'
    case _:
        Exception("Improper Waveguide Cmap Case")

wgNegAlpha_values = [c*li/wr for li in neg_Estd_BigList]
wgPosAlpha_values = [c*li/wr for li in pos_Estd_BigList]

negAlphaNorm = plt.Normalize(np.asarray([li.min() for li in wgNegAlpha_values]).min(),
                             np.asarray([li.max() for li in wgNegAlpha_values]).max())
posAlphaNorm = plt.Normalize(np.asarray([li.min() for li in wgPosAlpha_values]).min(),
                             np.asarray([li.max() for li in wgPosAlpha_values]).max())

posAlphaData = [posAlphaNorm(li) for li in wgPosAlpha_values]
negAlphaData = [negAlphaNorm(li) for li in wgNegAlpha_values]

dotSize = 75

for n in range(len(thetadegsList)):
    posScatter = ax.scatter(pos_evals_BigList[n], [thetadegsList[n] for i in pos_evals_BigList[n]], s=posAlphaData[n]*dotSize,
                            c=wgPosCmap_values[n], norm=wgNorm, cmap=wgCmap, alpha=posAlphaData[n])
    negScatter = ax.scatter(neg_evals_BigList[n], [thetadegsList[n] for i in neg_evals_BigList[n]], s=negAlphaData[n]*dotSize,
                            c=-wgNegCmap_values[n], norm=wgNorm, cmap=wgCmap, alpha=negAlphaData[n])

wgCbar = fig.colorbar(posScatter, ax=ax, orientation=wgColorbarOrientation)
wgCbar.set_label(wgColorbarLabel)

plt.ylim([-2, 40])
plt.xlim([1,6])
yticks = [0, 10, 20, 30, 40]
ylabels = [r'$0^\circ$',r'$10^\circ$',r'$20^\circ$',r'$30^\circ$',r'$40^\circ$']
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
plt.xlabel(r'Frequency [GHz]', size=18)
plt.ylabel(r'$\theta [^\circ]$', size=18)
plt.show()
#endregion