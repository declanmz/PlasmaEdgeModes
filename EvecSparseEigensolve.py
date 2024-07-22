import numpy as np
import pandas as pd
import scipy as scp
from tqdm import notebook
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex
from EigenSolveDegreeSaveJSON import BuildMatrix

def EvecSparseEigensolve_toJSON(file, save_directory, kmag_close, w_close_n, save_bool):
    #Define Constants
    e = 1.6e-19 #C - charge on electron
    m_e = 9.1094e-31 #kg - mass of electron
    mu_0 = 4e-7*np.pi #H/m - permeability of free space
    ep_0 = 8.854e-12 #F/m - permitivity of free space
    c = 3e8 #m/s - speed of light

    #Load JSON
    with open(file, 'r') as f:
        jsondata = json.load(f, object_hook=as_complex)
    
    thetadegs = jsondata['thetadegs']
    theta = jsondata['theta']
    wp0 = jsondata['wp0']
    wr = jsondata['wr']
    kzoffset = jsondata['kzoffset']
    N = jsondata['N']
    klist = wr * np.asarray(jsondata['klist_n'])/c
    evals_list_n = jsondata['evals_list_n']

    kId = np.argmin(np.abs(klist - kmag_close))
    kmag = klist[kId]
    evals_n = np.asarray(evals_list_n[kId])
    eval_n = evals_n[np.argmin(np.abs(evals_n - w_close_n))]

    ky = kmag * np.sin(theta)
    ky_n = c * ky / wr
    kz = kmag * np.cos(theta) + kzoffset
    kz_n = c * kz / wr

    # could make this a higher fidelity thing... but that would require changing wplist
    M = BuildMatrix(ky_n, kz_n, N, jsondata['wc_n'], jsondata['wplist_n'], 
                    jsondata['eplist'], jsondata['deltax_n'])
    eigsys = scp.sparse.linalg.eigs(M, k=1, sigma=eval_n, which='LM')
    
    eval_returned_n = eigsys[0][0]
    mixed_evec_n = eigsys[1][:,0]

    ux = [mixed_evec_n[9*i] for i in range(N)]
    uy = [mixed_evec_n[9*i + 1] for i in range(N)]
    uz = [mixed_evec_n[9*i + 2] for i in range(N)]
    Ex = [mixed_evec_n[9*i + 3] for i in range(N)]
    Ey = [mixed_evec_n[9*i + 4] for i in range(N)]
    Ez = [mixed_evec_n[9*i + 5] for i in range(N)]
    Bx_n = [mixed_evec_n[9*i + 6] for i in range(N)]
    By_n = [mixed_evec_n[9*i + 7] for i in range(N)]
    Bz_n = [mixed_evec_n[9*i + 8] for i in range(N)]

    evec_dict = {
        'sourceFile':file,
        'thetadegs':thetadegs,
        'theta':theta,
        'kzoffset':kzoffset,
        'kmag':kmag,
        'eval_fromSource_n':eval_n,
        'eval_returned_n':eval_returned_n,
        'Ex':Ex,
        'Ey':Ey,
        'Ez':Ez,
        'Bx_n':Bx_n,
        'By_n':By_n,
        'Bz_n':Bz_n,
        'ux':ux,
        'uy':uy,
        'uz':uz
    }

    if save_bool:
        dict_file = save_directory + f'/evec_{thetadegs}deg_{kmag:.5f}k0_{(eval_n.real):.5f}wr'.replace('.','_') + '.json'
        with open(dict_file, 'w') as f: 
            json.dump(evec_dict, f, cls=TypeEncoder)

    return evec_dict


# file = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/Testing/39deg_FilterA.json'
# save_directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/Testing'
# kmag_close = 300
# w_close_n = 3
# print(EvecSparseEigensolve_toJSON(file, save_directory, kmag_close, w_close_n, False))