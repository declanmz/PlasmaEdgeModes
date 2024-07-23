import scipy as scp
import numpy as np

x = 0
y = 0
z = 0

def func_zhat(phi,s,zp):
    return (-1*np.sin(phi)*(y-s*np.sin(phi)) + np.cos(phi)*(-x + s*np.cos(phi)))/(((x-s*np.cos(phi))**2 + (y-s*np.sin(phi))**2 + (z-zp)**2)**(3/2))

za = 1
zb = 2

def sa(z):
    return 1

def sb(z):
    return 2

def phia(s,z):
    return 0

def phib(s,z):
    return 2 * np.pi

B = scp.integrate.tplquad(func_zhat, za, zb, sa, sb, phia, phib)
print(B)