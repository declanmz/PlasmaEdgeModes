import numpy as np
import plotly.offline as plto
import plotly.graph_objects as go

import json
from math import radians

from JSONHelpers import TypeEncoder, as_complex
from EvecSparseSolve import EvecSparseSolve

# ----- Choose File -----
baseDirectory = 'C:/Users/decla/Documents/SPPL/PlasmaEdgeModes/EdgeCompV3/Setups'
setupFolder = 'Qin2023Linear'
kzoffset_n = 0.9
kyoffset_n = 0
thetadegs = 90

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

avgFilterBounds = [30, 70]
absAvgFilterBounds = [0, L_n]
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

#endregion


def MagPlasmaEigenmodes(wp, kx, ky, kz):
    H = np.array([[0,-1j,0,-1j*wp,0,0,0,0,0],
                  [1j,0,0,0,-1j*wp,0,0,0,0],
                  [0,0,0,0,0,-1j*wp,0,0,0],
                  [1j*wp,0,0,0,0,0,0,kz,-ky],
                  [0,1j*wp,0,0,0,0,-kz,0,kx],
                  [0,0,1j*wp,0,0,0,ky,-kx,0],
                  [0,0,0,0,-kz,ky,0,0,0],
                  [0,0,0,kz,0,-kx,0,0,0],
                  [0,0,0,-ky,kx,0,0,0,0]])
    omn, fn = np.linalg.eigh(H)
    return [omn, fn]

def draw_surfaces(fig, wp, points, kperpmin, kperpmax, kzmin, kzmax, maxnum=4, colorList=['blue', 'red', 'green', 'orange']):
    kperp_points = np.linspace(kperpmin, kperpmax, points)
    kz_points = np.linspace(kzmin, kzmax, points)
    mesh_kperp, mesh_kz = np.meshgrid(kperp_points, kz_points)
    flat_kperp = mesh_kperp.ravel()
    flat_kz = mesh_kz.ravel()
    flat_surfaces = [np.zeros(flat_kperp.shape), np.zeros(flat_kperp.shape), np.zeros(flat_kperp.shape), np.zeros(flat_kperp.shape)]

    for i in range(len(flat_kperp)):
        oms = np.sort(MagPlasmaEigenmodes(wp, flat_kperp[i], 0, flat_kz[i])[0])
        for j in range(maxnum):
            flat_surfaces[j][i] = oms[j + 5]

    mesh_surfaces = [flat_surface.reshape(mesh_kperp.shape) for flat_surface in flat_surfaces]

    # Plot the surfaces
    for i in range(maxnum):
        fig.add_trace(go.Surface(x=mesh_kperp, y=mesh_kz, z=mesh_surfaces[i], colorscale=[[0, colorList[i]], [1, colorList[i]]], showscale=False, opacity=0.1))

    kplus = wp / np.sqrt(1 + wp)
    if wp < 1:
        kminus = wp / np.sqrt(1 - wp)
    else:
        kminus = np.inf

    # Plot scatter points for critical values
    critical_points = []
    if kplus > kzmin and kplus < kzmax:
        critical_points.append((0, kplus, wp))
    if -kplus > kzmin and -kplus < kzmax:
        critical_points.append((0, -kplus, wp))
    if kminus > kzmin and kminus < kzmax:
        critical_points.append((0, kminus, wp))
    if -kminus > kzmin and -kminus < kzmax:
        critical_points.append((0, -kminus, wp))

    for point in critical_points:
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', marker=dict(color='black', size=5)))



# Initialize Plotly figure
fig = go.Figure()

for n in range(len(klist_n)):
    fig.add_trace(go.Scatter3d(
        x=[klist_n[n]] * len(evals_n_list[n]),  # Repeating klist_n[n] for each eval
        y=[kzoffset_n] * len(evals_n_list[n]),  # Repeating kzoffset_n for each eval
        z=evals_n_list[n].real,
        mode='markers',  # Show only markers, no lines
        marker=dict(color='red', size=1)
    ))
        # dispScatter = axs['e'].scatter([klist_n[n] for i in evals_n_list[n]], evals_n_list[n].real, s=dotsize, 
        #                 c=dispCmap_values[n], norm=dispCmap_norm, cmap=dispCmap)

# Generate data for the surfaces
wp1 = 0.8
points = 100
kperpmin = -2
kperpmax = 2
kzmin = -2
kzmax = 2
maxnum = 2

# Draw surfaces for the given wp values
draw_surfaces(fig, wp1, points, kperpmin, kperpmax, kzmin, kzmax, maxnum=maxnum, colorList=['green', 'green'])

wp2 = 0.45
draw_surfaces(fig, wp2, points, kperpmin, kperpmax, kzmin, kzmax, maxnum=maxnum, colorList=['yellow', 'yellow'])


# Set axis labels and title
fig.update_layout(
    title='Bulk Modes Dispersion Surfaces',
    scene=dict(
        aspectmode='manual',  # Use manual aspect mode to control the ratios
        aspectratio=dict(x=1, y=1, z=0.8),  # Increase the z value to make the z-axis larger
        xaxis_title='kperp',
        yaxis_title='kz',
        zaxis_title='w'
    )
)

plot_url = plto.plot(fig, filename='BulkModesDispersion.html', include_mathjax='cdn')
