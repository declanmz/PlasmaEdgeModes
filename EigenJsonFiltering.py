import numpy as np
import pandas as pd
import scipy as scp
import csv
from tqdm import tqdm
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex

def filterEigJSON(file, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs):
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

    evals_list = list(map(np.array, jsondata['evals_list']))
    Eavg_list = list(map(np.array, jsondata['Eavg_list']))
    Estd_list = list(map(np.array, jsondata['Estd_list']))
    Emax_list = list(map(np.array, jsondata['Emax_list']))

    # ---------- Filtering ----------
    for i in range(jsondata['Nk']):
        EavgFilteredIndices = np.where((Eavg_list[i].real < filter_posAvgBound) & (Eavg_list[i].real > filter_negAvgBound))
        EstdFilteredIndices = np.where((Estd_list[i].real < filter_maxStd))
        EmaxFilteredIndices = np.where((Emax_list[i].real > filter_EmaxMin))
        filteredIndices = np.intersect1d(np.intersect1d(EavgFilteredIndices, EstdFilteredIndices), EmaxFilteredIndices)

        jsondata['evals_list'][i] = evals_list[i][filteredIndices]
        jsondata['Eavg_list'][i] = Eavg_list[i][filteredIndices]
        jsondata['Estd_list'][i] = Estd_list[i][filteredIndices]
        jsondata['Emax_list'][i] = Emax_list[i][filteredIndices]


    # ---------- Save As New JSON ----------
    jsondata['filter_posAvgBound'] = filter_posAvgBound
    jsondata['filter_negAvgBound'] = filter_negAvgBound
    jsondata['filter_maxStd'] = filter_maxStd
    jsondata['filter_EmaxMin'] = filter_EmaxMin

    dict_file = directory + f'/{thetadegs}deg_{filterName}.json'
    with open(dict_file, 'w') as f: 
        json.dump(jsondata, f, cls=TypeEncoder)

    print('Filtering Done.')


# directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/SetupOG'
# thetadegs = 38
# file = directory + f'/{thetadegs}deg_CutSort.json'

# filterName = 'FilterB'
# filter_posAvgBound = 15e-3
# filter_negAvgBound = 0
# filter_maxStd = 1 #4e-3
# filter_EmaxMin = 1e-5

# filterEigJSON(file, filterName, directory, filter_posAvgBound, filter_negAvgBound, filter_maxStd, filter_EmaxMin, thetadegs)