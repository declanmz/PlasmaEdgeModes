import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox

# Create the figure and adjust its size
fig = plt.figure(figsize=(10, 6))

# Create the 3D plot on the left side
ax = plt.subplot2grid((1, 2), (0, 0), projection='3d')

# Set initial bounds for the axes
bounds = [-2, 2, -2, 2, 0, 3]
wp = 0.8
points = 20

ax.set_xlim(bounds[0],bounds[1])
ax.set_ylim(bounds[2],bounds[3])
ax.set_zlim(bounds[4],bounds[5])

# Labels


# Create textbox axes for x, y, and z limits on the right side
ax_text_xmin = plt.axes([0.6, 0.7, 0.1, 0.05])
ax_text_xmax = plt.axes([0.8, 0.7, 0.1, 0.05])
ax_text_ymin = plt.axes([0.6, 0.6, 0.1, 0.05])
ax_text_ymax = plt.axes([0.8, 0.6, 0.1, 0.05])
ax_text_zmin = plt.axes([0.6, 0.5, 0.1, 0.05])
ax_text_zmax = plt.axes([0.8, 0.5, 0.1, 0.05])

# Create text boxes
text_xmin = TextBox(ax_text_xmin, r'$k_\perp$ Min', initial=str(bounds[0]))
text_xmax = TextBox(ax_text_xmax, r'$k_\perp$ Max', initial=str(bounds[1]))
text_ymin = TextBox(ax_text_ymin, r'$k_z$ Min', initial=str(bounds[2]))
text_ymax = TextBox(ax_text_ymax, r'$k_z$ Max', initial=str(bounds[3]))
text_zmin = TextBox(ax_text_zmin, r'$\omega/\Omega$ Min', initial=str(bounds[4]))
text_zmax = TextBox(ax_text_zmax, r'$\omega/\Omega$ Max', initial=str(bounds[5]))

# Create slider axes for the z value of the red dot
ax_slider_wp = plt.axes([0.6, 0.3, 0.3, 0.03], facecolor='lightgoldenrodyellow')
slider_wp = Slider(ax_slider_wp, r'$\omega_p/\Omega$', 0, 2, valinit=wp)

# Create a textbox to manually set the z value of the red dot
ax_text_wp_value = plt.axes([0.6, 0.2, 0.2, 0.05])
text_wp_value = TextBox(ax_text_wp_value, r'$\omega_p/\Omega$', initial=f"{wp:.2f}")

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

def draw_surfaces(wp, points, kperpmin, kperpmax, kzmin, kzmax, wmin, wmax):
    ax.cla()

    ax.set_xlabel(r'$k_\perp$')
    ax.set_ylabel(r'$k_z$')
    ax.set_zlabel(r'$\omega/\Omega$')
    ax.set_title('Cold Plasma Bulk Dispersion Surfaces')

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

    # Plot the surfaces
    for i in range(4):
        ax.plot_surface(mesh_kperp, mesh_kz, mesh_surfaces[i], alpha=0.6, edgecolor='none')

    mesh_lightcone = np.sqrt(mesh_kperp**2 + mesh_kz**2)

    ax.plot_surface(mesh_kperp, mesh_kz, mesh_lightcone, alpha=0.1, edgecolor='none', color='yellow')

    kplus = wp/np.sqrt(1+wp)
    if wp < 1:
        kminus = wp/np.sqrt(1-wp)
    else:
        kminus = np.inf
    if kplus > kzmin and kplus < kzmax:
        ax.scatter(0, kplus,wp, color='red')
    if -kplus > kzmin and -kplus < kzmax:
        ax.scatter(0, -kplus,wp, color='blue')
    if kminus > kzmin and kminus < kzmax:
        ax.scatter(0, kminus,wp, color='blue')
    if -kminus > kzmin and -kminus < kzmax:
        ax.scatter(0, -kminus,wp, color='red')

    ax.set_xlim(kperpmin, kperpmax)
    ax.set_ylim(kzmin, kzmax)
    ax.set_zlim(wmin, wmax)

# Update function to adjust the bounds based on the textbox values
def update_bounds(val):
    try:
        bounds = [float(text_xmin.text), float(text_xmax.text), float(text_ymin.text), 
                  float(text_ymax.text), float(text_zmin.text), float(text_zmax.text)]
        draw_surfaces(wp, points, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        fig.canvas.draw_idle()
    except ValueError:
        pass  # If the input is not a valid float, ignore it

update_bounds(0)

# Update function to move the red dot based on the slider value
def update_point(val):
    wp = slider_wp.val
    text_wp_value.set_val(f"{wp:.2f}")  # Update the text box to match the slider with 2 decimal places
    draw_surfaces(wp, points, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
    fig.canvas.draw_idle()
 
# Update function to override the slider with the textbox value
def update_z_from_text(val):
    try:
        wp = float(text_wp_value.text)
        slider_wp.set_val(wp)  # Update the slider to match the text box
        draw_surfaces(wp, points, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        fig.canvas.draw_idle()
    except ValueError:
        pass  # Ignore invalid inputs

# Connect text boxes and slider to their respective update functions
text_xmin.on_submit(update_bounds)
text_xmax.on_submit(update_bounds)
text_ymin.on_submit(update_bounds)
text_ymax.on_submit(update_bounds)
text_zmin.on_submit(update_bounds)
text_zmax.on_submit(update_bounds)

slider_wp.on_changed(update_point)
text_wp_value.on_submit(update_z_from_text)

ax.invert_yaxis()

plt.show()
