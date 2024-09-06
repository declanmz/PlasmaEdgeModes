import numpy as np

from Solve2dSlice_ToJSON import Solve2dSlice_ToJSON_dimensioned, Solve2dSlice_ToJSON_normalized

# ---------- Define Constants ----------
e = 1.6e-19 #C - charge on electron
m_e = 9.1094e-31 #kg - mass of electron
mu_0 = 4e-7*np.pi #H/m - permeability of free space
ep_0 = 8.854e-12 #F/m - permitivity of free space
c = 3e8 #m/s - speed of light

# ---------- Setup parameters ----------
directory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/EdgeCompV3/Setups/Qin2023Linear'
fr = 5e9
wc_n = 1
L_n = 100
N = 100
wp1_n = 0.8
wp2_n = 0.45

def wp_n(x):
    if np.abs(x) < 30:
        return wp1_n
    elif np.abs(x) < 70:
        return wp2_n + (wp1_n-wp2_n)*(70-np.abs(x))/40
    else:
        return wp2_n

def ep(x):
    return 1

kmin_n = -2
kmax_n = 2
thetadegs = 90
Nk = 50
kzoffset_n = 0
kyoffset_n = 0

for kz_n in [0.6, 1.2, 1.4]:
    Solve2dSlice_ToJSON_normalized(directory, fr, wc_n, L_n, N, wp_n, ep, kmin_n, kmax_n, thetadegs, Nk, kz_n, kyoffset_n)