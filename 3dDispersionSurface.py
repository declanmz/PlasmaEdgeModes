from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
    #fn = (vn, En, Bn), 9x1 vector
    return [omn, fn]

defaultColors = ['C0', 'C1','C2', 'C3']

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw_surfaces(wp, points, kperpmin, kperpmax, kzmin, kzmax, maxnum=4, colorList=defaultColors):
    kperp_points = np.linspace(kperpmin, kperpmax, points)
    kz_points = np.linspace(kzmin, kzmax, points)
    mesh_kperp, mesh_kz = np.meshgrid(kperp_points, kz_points)
    flat_kperp = mesh_kperp.ravel()
    flat_kz = mesh_kz.ravel()
    flat_surfaces = [np.zeros(flat_kperp.shape),np.zeros(flat_kperp.shape),np.zeros(flat_kperp.shape),np.zeros(flat_kperp.shape)]

    for i in range(len(flat_kperp)):
        oms = np.sort(MagPlasmaEigenmodes(wp, flat_kperp[i], 0, flat_kz[i])[0])
        for j in range(4):
            flat_surfaces[j][i] = oms[j+5]

    mesh_surfaces = [flat_surface.reshape(mesh_kperp.shape) for flat_surface in flat_surfaces]

    # Plot the surface
    for i in range(maxnum):
        ax.plot_surface(mesh_kperp, mesh_kz, mesh_surfaces[i], alpha=0.5, edgecolor='none', color=colorList[i])

    kplus = wp/np.sqrt(1+wp)
    if wp < 1:
        kminus = wp/np.sqrt(1-wp)
    else:
        kminus = np.inf
    if kplus > kzmin and kplus < kzmax:
        ax.scatter(0, kplus,wp, color='black')
    if -kplus > kzmin and -kplus < kzmax:
        ax.scatter(0, -kplus,wp, color='black')
    if kminus > kzmin and kminus < kzmax:
        ax.scatter(0, kminus,wp, color='black')
    if -kminus > kzmin and -kminus < kzmax:
        ax.scatter(0, -kminus,wp, color='black')


# Generate data for the surface
wp = 0.8
points = 50
kperpmin = -2
kperpmax = 2
kzmin = -2
kzmax = 2
maxnum = 2

draw_surfaces(wp, points, kperpmin, kperpmax, kzmin, kzmax, maxnum=maxnum, colorList=['green', 'green'])

wp3 = 0.45
draw_surfaces(wp3, points, kperpmin, kperpmax, kzmin, kzmax, maxnum=maxnum, colorList=['yellow','yellow'])


plt.gca().invert_xaxis()

# Labels
ax.set_xlabel('kperp')
ax.set_ylabel('kz')
ax.set_zlabel('w')
ax.set_title('dispersion surfaces')

plt.show()