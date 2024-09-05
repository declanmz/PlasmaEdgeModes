import numpy as np
import pandas as pd
import scipy as scp
from tqdm import notebook
import cmath
import json
from math import radians

from JSONHelpers import TypeEncoder, as_complex
from Solve2dSlice_ToJSON import BuildMatrix

def EvecSparseSolve(file, kmag_close_n, w_close_n):

    with open(file, 'r') as f:
        jsondata = json.load(f, object_hook=as_complex)
    
    thetadegs = jsondata['thetadegs']
    kzoffset_n = jsondata['kzoffset_n']
    kyoffset_n = jsondata['kyoffset_n']
    N = jsondata['N']
    klist_n = np.asarray(jsondata['klist_n'])
    evals_n_list = jsondata['evals_n_list']

    kId = np.argmin(np.abs(klist_n - kmag_close_n))
    kmag_n = klist_n[kId]
    evals_n = np.asarray(evals_n_list[kId])
    eval_n = evals_n[np.argmin(np.abs(evals_n - w_close_n))]

    ky_n = kmag_n * np.sin(radians(thetadegs)) + kyoffset_n
    kz_n = kmag_n * np.cos(radians(thetadegs)) + kzoffset_n

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
        'kzoffset_n':kzoffset_n,
        'kyoffset_n':kyoffset_n,
        'kmag_n':kmag_n,
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

    # if save_bool:
    #     dict_file = save_directory + f'/evec_{thetadegs}deg_{kmag:.5f}k0_{(eval_n.real):.5f}wr'.replace('.','_') + '.json'
    #     with open(dict_file, 'w') as f: 
    #         json.dump(evec_dict, f, cls=TypeEncoder)

    return evec_dict