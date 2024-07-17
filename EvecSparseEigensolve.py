import numpy as np
import pandas as pd
import scipy as scp
from tqdm import notebook
import cmath
import json

from JSONHelpers import TypeEncoder, as_complex
from EigenSolveDegreeSaveJSON import BuildMatrix

def EvecSparseEigensolve_toJSON(file, save_directory, k0mag_close, w0_close):
    #Define Constants
    e = 1.6e-19 #C - charge on electron
    m_e = 9.1094e-31 #kg - mass of electron
    al = m_e/e #used in matrix
    alin = 1/al
    mu_0 = 4e-7*np.pi #H/m - permeability of free space
    ep_0 = 8.854e-12 #F/m - permitivity of free space
    c = 3e8 #m/s - speed of light

    #Load JSON
    with open(file, 'r') as f:
        jsondata = json.load(f, object_hook=as_complex)
    
    thetadegs = jsondata['thetadegs']
    theta = jsondata['theta']
    wp0 = jsondata['wp0']
    kzoffset = jsondata['kzoffset']
    N = jsondata['N']
    k0list = np.asarray(jsondata['k0list'])
    evals_list = jsondata['evals_list']

    k0Id = np.argmin(np.abs(k0list - k0mag_close))
    k0mag = k0list[k0Id]
    evals = np.asarray(evals_list[k0Id])
    w_close = w0_close * wp0
    eval = evals[np.argmin(np.abs(evals - w_close))]

    kmag = wp0 * k0mag / c
    ky = kmag * np.sin(theta)
    kz = kmag * np.cos(theta) + kzoffset

    # could make this a higher fidelity thing... but that would require changing wplist, etc...
    M = BuildMatrix(ky, kz, N, jsondata['B0'], jsondata['wplist'], 
                    jsondata['eplist'], jsondata['mulist'], jsondata['deltax'])
    eigsys = scp.sparse.linalg.eigs(M, k=1, sigma=eval, which='LM')
    
    eval_returned = eigsys[0][0]
    mixed_evec = eigsys[1][:,0]

    vx = [mixed_evec[9*i] for i in range(N)]
    vy = [mixed_evec[9*i + 1] for i in range(N)]
    vz = [mixed_evec[9*i + 2] for i in range(N)]
    Ex = [mixed_evec[9*i + 3] for i in range(N)]
    Ey = [mixed_evec[9*i + 4] for i in range(N)]
    Ez = [mixed_evec[9*i + 5] for i in range(N)]
    Bx = [mixed_evec[9*i + 6] for i in range(N)]
    By = [mixed_evec[9*i + 7] for i in range(N)]
    Bz = [mixed_evec[9*i + 8] for i in range(N)]

    evec_dict = {
        'sourceFile':file,
        'thetadegs':thetadegs,
        'theta':theta,
        'kzoffset':kzoffset,
        'k0mag':k0mag,
        'eval_fromSource':eval,
        'eval_returned':eval_returned,
        'Ex':Ex,
        'Ey':Ey,
        'Ez':Ez,
        'Bx':Bx,
        'By':By,
        'Bz':Bz,
        'vx':vx,
        'vy':vy,
        'vz':vz
    }

    dict_file = save_directory + f'/evec_{thetadegs}deg_{k0mag:.5f}k0_{(eval.real/wp0):.5f}w0'.replace('.','_') + '.json'
    with open(dict_file, 'w') as f: 
        json.dump(evec_dict, f, cls=TypeEncoder)


# file = './SetupOG_ReducedRange/38deg_FilterA.json'
# save_directory = './Testing'
# print(EvecSparseEigensolve_toJSON(file, save_directory, -0.6, 0.3))