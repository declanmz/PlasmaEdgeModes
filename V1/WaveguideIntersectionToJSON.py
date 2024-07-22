import numpy as np
import pandas as pd
import scipy as scp
import csv
from tqdm import tqdm
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex

def findWgIntersection_UpdateJSON(file, wgEpsilon, thetadegs):
    # ---------- Define Constants ----------
    e = 1.6e-19 #C - charge on electron
    m_e = 9.1094e-31 #kg - mass of electron
    al = m_e/e #define constant used in matrix
    alin = 1/al #al inverse
    mu_0 = 4e-7*np.pi #H/m - permeability of free space
    ep_0 = 8.854e-12 #F/m - permitivity of free space
    c = 3e8 #m/s - speed of light

    # ---------- Read JSON File ----------
    with open(file, 'r') as f:
        jsondata = json.load(f, object_hook=as_complex)

    thetadegsJSON = jsondata['thetadegs']
    if thetadegs != thetadegsJSON:
        raise Exception("Thetadegs mismatch")

    wp0 = jsondata['wp0']
    k0list = np.asarray(jsondata['k0list'])
    evals_list = list(map(np.array, jsondata['evals_list']))
    Eavg_list = list(map(np.array, jsondata['Eavg_list']))
    Estd_list = list(map(np.array, jsondata['Estd_list']))
    Emax_list = list(map(np.array, jsondata['Emax_list']))

    # Waveguide dispersion
    dwg = 6.25e-3
    # wpwg = 2 * np.pi * 3.5e9
    wpwg = wp0
    kpwg = wpwg/c
    dispersionload = np.loadtxt("C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/WaveguideDispersion/eps_4.txt")
    k0waveguide = dispersionload[:,0]/180 /dwg * np.pi / kpwg
    w_waveguide = dispersionload[:,1] * 2 * np.pi * 1e9

    # Interpolate to k0list **negative side**
    # kinda stupid to be splitting these up and have duplicate code, but it's useful later
    # Overall this is just bad code, but whatever it was the fastest to write

    neg_k0Ids = np.where(k0list < 0)[0].tolist()
    neg_k0list = k0list[neg_k0Ids]
    neg_evals_list = [evals_list[i] for i in neg_k0Ids]
    neg_Eavg_list = [Eavg_list[i] for i in neg_k0Ids]
    neg_Estd_list = [Estd_list[i] for i in neg_k0Ids]
    neg_Emax_list = [Emax_list[i] for i in neg_k0Ids]

    neg_wgInterpolation = np.interp(neg_k0list, (-k0waveguide)[::-1], w_waveguide[::-1])

    neg_wgDistance = np.array([])
    neg_wgEval = np.array([])
    neg_wgEavg = np.array([])
    neg_wgEstd = np.array([])
    neg_wgEmax = np.array([])

    for i in range(len(neg_k0list)):
        closestEvalId = np.argmin(np.abs(neg_evals_list[i].real-neg_wgInterpolation[i]))
        neg_wgDistance = np.append(neg_wgDistance, np.abs(neg_evals_list[i][closestEvalId].real - neg_wgInterpolation[i]))
        neg_wgEval = np.append(neg_wgEval, neg_evals_list[i][closestEvalId].real)
        neg_wgEavg = np.append(neg_wgEavg, neg_Eavg_list[i][closestEvalId].real)
        neg_wgEstd = np.append(neg_wgEstd, neg_Estd_list[i][closestEvalId].real)
        neg_wgEmax = np.append(neg_wgEmax, neg_Emax_list[i][closestEvalId].real)

    neg_epsilonIndices = np.where(neg_wgDistance < wgEpsilon)
    neg_epsilon_k0list = neg_k0list[neg_epsilonIndices]
    neg_epsilon_wgDistance = neg_wgDistance[neg_epsilonIndices]
    neg_epsilon_wgEval = neg_wgEval[neg_epsilonIndices]
    neg_epsilon_wgEavg = neg_wgEavg[neg_epsilonIndices]
    neg_epsilon_wgEstd = neg_wgEstd[neg_epsilonIndices]
    neg_epsilon_wgEmax = neg_wgEmax[neg_epsilonIndices]

    # Interpolate to k0list **positive side**
    pos_k0Ids = np.where(k0list > 0)[0].tolist()
    pos_k0list = k0list[pos_k0Ids]
    pos_evals_list = [evals_list[i] for i in pos_k0Ids]
    pos_Eavg_list = [Eavg_list[i] for i in pos_k0Ids]
    pos_Estd_list = [Estd_list[i] for i in pos_k0Ids]
    pos_Emax_list = [Emax_list[i] for i in pos_k0Ids]

    pos_wgInterpolation = np.interp(pos_k0list, k0waveguide, w_waveguide)

    pos_wgDistance = np.array([])
    pos_wgEval = np.array([])
    pos_wgEavg = np.array([])
    pos_wgEstd = np.array([])
    pos_wgEmax = np.array([])

    for i in range(len(pos_k0list)):
        closestEvalId = np.argmin(np.abs(pos_evals_list[i].real-pos_wgInterpolation[i]))
        pos_wgDistance = np.append(pos_wgDistance, np.abs(pos_evals_list[i][closestEvalId].real - pos_wgInterpolation[i]))
        pos_wgEval = np.append(pos_wgEval, pos_evals_list[i][closestEvalId].real)
        pos_wgEavg = np.append(pos_wgEavg, pos_Eavg_list[i][closestEvalId].real)
        pos_wgEstd = np.append(pos_wgEstd, pos_Estd_list[i][closestEvalId].real)
        pos_wgEmax = np.append(pos_wgEmax, pos_Emax_list[i][closestEvalId].real)

    pos_epsilonIndices = np.where(pos_wgDistance < wgEpsilon)
    pos_epsilon_k0list = pos_k0list[pos_epsilonIndices]
    pos_epsilon_wgDistance = pos_wgDistance[pos_epsilonIndices]
    pos_epsilon_wgEval = pos_wgEval[pos_epsilonIndices]
    pos_epsilon_wgEavg = pos_wgEavg[pos_epsilonIndices]
    pos_epsilon_wgEstd = pos_wgEstd[pos_epsilonIndices]
    pos_epsilon_wgEmax = pos_wgEmax[pos_epsilonIndices]

    # ---------- Update JSON File ----------
    jsondata['wgEpsilon'] = wgEpsilon

    jsondata['wgNeg_k0list'] = neg_epsilon_k0list
    jsondata['wgNeg_distanceList'] = neg_epsilon_wgDistance
    jsondata['wgNeg_evalList'] = neg_epsilon_wgEval
    jsondata['wgNeg_EavgList'] = neg_epsilon_wgEavg
    jsondata['wgNeg_EstdList'] = neg_epsilon_wgEstd
    jsondata['wgNeg_EmaxList'] = neg_epsilon_wgEmax

    jsondata['wgPos_k0list'] = pos_epsilon_k0list
    jsondata['wgPos_distanceList'] = pos_epsilon_wgDistance
    jsondata['wgPos_evalList'] = pos_epsilon_wgEval
    jsondata['wgPos_EavgList'] = pos_epsilon_wgEavg
    jsondata['wgPos_EstdList'] = pos_epsilon_wgEstd
    jsondata['wgPos_EmaxList'] = pos_epsilon_wgEmax


    with open(file, 'w') as f: 
        json.dump(jsondata, f, cls=TypeEncoder)

    print('Waveguide Intersection Done.')


# #execute code for particular file
# directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/SetupOG'
# thetadegs = 38
# filterName = 'FilterB'
# file = directory + f'/{thetadegs}deg_{filterName}.json'

# wgEpsilon = 1e9
# findWgIntersection_UpdateJSON(file, wgEpsilon, thetadegs)