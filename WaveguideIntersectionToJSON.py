import numpy as np
import pandas as pd
import scipy as scp
import csv
from tqdm import tqdm
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex

def findWgIntersection_UpdateJSON(file, wgEpsilon_n, thetadegs):
    # ---------- Define Constants ----------
    e = 1.6e-19 #C - charge on electron
    m_e = 9.1094e-31 #kg - mass of electron
    mu_0 = 4e-7*np.pi #H/m - permeability of free space
    ep_0 = 8.854e-12 #F/m - permitivity of free space
    c = 3e8 #m/s - speed of light

    # ---------- Read JSON File ----------
    with open(file, 'r') as f:
        jsondata = json.load(f, object_hook=as_complex)

    thetadegsJSON = jsondata['thetadegs']
    if thetadegs != thetadegsJSON:
        raise Exception("Thetadegs mismatch")

    wr = jsondata['wr']
    klist_n = np.asarray(jsondata['klist_n'])
    klist = wr*klist_n/c
    evals_list_n = list(map(np.array, jsondata['evals_list_n']))
    Eavg_list_n = list(map(np.array, jsondata['Eavg_list_n']))
    Estd_list_n = list(map(np.array, jsondata['Estd_list_n']))
    Emax_list_n = list(map(np.array, jsondata['Emax_list_n']))

    # Waveguide dispersion
    dwg = 6.25e-3
    dispersionload = np.loadtxt("C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/WaveguideDispersion/eps_4.txt")
    k_waveguide = dispersionload[:,0]/180 /dwg * np.pi
    w_waveguide_n = dispersionload[:,1] * 2 * np.pi * 1e9 / wr

    # Interpolate to klist **negative side**
    # kinda stupid to be splitting these up and have duplicate code, but it's useful later
    # Overall this is just bad code, but whatever it was the fastest to write

    neg_k0Ids = np.where(klist < 0)[0].tolist()
    neg_klist = klist[neg_k0Ids]
    neg_evals_list_n = [evals_list_n[i] for i in neg_k0Ids]
    neg_Eavg_list_n = [Eavg_list_n[i] for i in neg_k0Ids]
    neg_Estd_list_n = [Estd_list_n[i] for i in neg_k0Ids]
    neg_Emax_list_n = [Emax_list_n[i] for i in neg_k0Ids]

    neg_wgInterpolation = np.interp(neg_klist, (-k_waveguide)[::-1], w_waveguide_n[::-1])

    neg_wgDistance = np.array([])
    neg_wgEval_n = np.array([])
    neg_wgEavg_n = np.array([])
    neg_wgEstd_n = np.array([])
    neg_wgEmax_n = np.array([])

    for i in range(len(neg_klist)):
        closestEvalId = np.argmin(np.abs(neg_evals_list_n[i].real-neg_wgInterpolation[i]))
        neg_wgDistance = np.append(neg_wgDistance, np.abs(neg_evals_list_n[i][closestEvalId].real - neg_wgInterpolation[i]))
        neg_wgEval_n = np.append(neg_wgEval_n, neg_evals_list_n[i][closestEvalId].real)
        neg_wgEavg_n = np.append(neg_wgEavg_n, neg_Eavg_list_n[i][closestEvalId].real)
        neg_wgEstd_n = np.append(neg_wgEstd_n, neg_Estd_list_n[i][closestEvalId].real)
        neg_wgEmax_n = np.append(neg_wgEmax_n, neg_Emax_list_n[i][closestEvalId].real)

    neg_epsilonIndices = np.where(neg_wgDistance < wgEpsilon_n)
    neg_epsilon_klist = neg_klist[neg_epsilonIndices]
    neg_epsilon_wgDistance = neg_wgDistance[neg_epsilonIndices]
    neg_epsilon_wgEval_n = neg_wgEval_n[neg_epsilonIndices]
    neg_epsilon_wgEavg_n = neg_wgEavg_n[neg_epsilonIndices]
    neg_epsilon_wgEstd_n = neg_wgEstd_n[neg_epsilonIndices]
    neg_epsilon_wgEmax_n = neg_wgEmax_n[neg_epsilonIndices]

    # Interpolate to klist **positive side**
    pos_k0Ids = np.where(klist > 0)[0].tolist()
    pos_klist = klist[pos_k0Ids]
    pos_evals_list_n = [evals_list_n[i] for i in pos_k0Ids]
    pos_Eavg_list_n = [Eavg_list_n[i] for i in pos_k0Ids]
    pos_Estd_list_n = [Estd_list_n[i] for i in pos_k0Ids]
    pos_Emax_list_n = [Emax_list_n[i] for i in pos_k0Ids]

    pos_wgInterpolation = np.interp(pos_klist, k_waveguide, w_waveguide_n)

    pos_wgDistance = np.array([])
    pos_wgEval_n = np.array([])
    pos_wgEavg_n = np.array([])
    pos_wgEstd_n = np.array([])
    pos_wgEmax_n = np.array([])

    for i in range(len(pos_klist)):
        closestEvalId = np.argmin(np.abs(pos_evals_list_n[i].real-pos_wgInterpolation[i]))
        pos_wgDistance = np.append(pos_wgDistance, np.abs(pos_evals_list_n[i][closestEvalId].real - pos_wgInterpolation[i]))
        pos_wgEval_n = np.append(pos_wgEval_n, pos_evals_list_n[i][closestEvalId].real)
        pos_wgEavg_n = np.append(pos_wgEavg_n, pos_Eavg_list_n[i][closestEvalId].real)
        pos_wgEstd_n = np.append(pos_wgEstd_n, pos_Estd_list_n[i][closestEvalId].real)
        pos_wgEmax_n = np.append(pos_wgEmax_n, pos_Emax_list_n[i][closestEvalId].real)

    pos_epsilonIndices = np.where(pos_wgDistance < wgEpsilon_n)
    pos_epsilon_klist = pos_klist[pos_epsilonIndices]
    pos_epsilon_wgDistance = pos_wgDistance[pos_epsilonIndices]
    pos_epsilon_wgEval_n = pos_wgEval_n[pos_epsilonIndices]
    pos_epsilon_wgEavg_n = pos_wgEavg_n[pos_epsilonIndices]
    pos_epsilon_wgEstd_n = pos_wgEstd_n[pos_epsilonIndices]
    pos_epsilon_wgEmax_n = pos_wgEmax_n[pos_epsilonIndices]

    # ---------- Update JSON File ----------
    jsondata['wgEpsilon_n'] = wgEpsilon_n

    jsondata['wgNeg_klist'] = neg_epsilon_klist
    jsondata['wgNeg_distanceList'] = neg_epsilon_wgDistance
    jsondata['wgNeg_evalList_n'] = neg_epsilon_wgEval_n
    jsondata['wgNeg_EavgList_n'] = neg_epsilon_wgEavg_n
    jsondata['wgNeg_EstdList_n'] = neg_epsilon_wgEstd_n
    jsondata['wgNeg_EmaxList_n'] = neg_epsilon_wgEmax_n

    jsondata['wgPos_klist'] = pos_epsilon_klist
    jsondata['wgPos_distanceList'] = pos_epsilon_wgDistance
    jsondata['wgPos_evalList_n'] = pos_epsilon_wgEval_n
    jsondata['wgPos_EavgList_n'] = pos_epsilon_wgEavg_n
    jsondata['wgPos_EstdList_n'] = pos_epsilon_wgEstd_n
    jsondata['wgPos_EmaxList_n'] = pos_epsilon_wgEmax_n


    with open(file, 'w') as f: 
        json.dump(jsondata, f, cls=TypeEncoder)

    print('Waveguide Intersection Done.')


#execute code for particular file
directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/Testing'
thetadegs = 39
filterName = 'FilterA'
file = directory + f'/{thetadegs}deg_{filterName}.json'

wgEpsilon = 1
findWgIntersection_UpdateJSON(file, wgEpsilon, thetadegs)