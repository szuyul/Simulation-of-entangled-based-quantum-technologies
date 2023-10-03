import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ref: https://iopscience.iop.org/article/10.1088/1742-6596/1380/1/012023/pdf
# video: https://www.youtube.com/watch?v=1MaOqvnkBxk&t=408s

def refractive_index(oe, wavelength, angle):
    # BBO
    # wavelength unit: um
    n_o = np.sqrt( 2.7359 + np.divide(0.01878, np.square(wavelength) - 0.01822) + 0.01354 * np.square(wavelength) )
    n_e = np.sqrt( 2.3753 + np.divide(0.01224, np.square(wavelength) - 0.01667) + 0.01516 * np.square(wavelength) )

    n = n_o if oe == 'o' else np.divide(1, np.sqrt( np.square(np.cos(angle)/n_o) + np.square(np.sin(angle)/n_e) ))
    return n

def emission_angle(wavelength, OA_angle):
    eqn = lambda x: abs(refractive_index('o', 2*wavelength/1e3, 0)*np.cos(x) - refractive_index('e', wavelength/1e3, OA_angle))
    res = minimize(eqn, 0, method = 'Nelder-Mead', tol=1e-6)
    return res.x[0]

class entangled_photons():
    def __init__(self, wavelength, OA_angle):
        self.wavelength = wavelength
        self.OA_angle = OA_angle
        self.theta = emission_angle(self.wavelength, self.OA_angle)
        self.phi = 2*np.pi*np.random.uniform(0, 1)
        self.polarization = np.random.binomial(1, 0.5, 1) # 0 is H, 1 is V
    



#%% refractive index of nonlinear crystal
wavelength = np.linspace(300, 1100) # unit: nm
angle = np.pi/6

_, ax1 = plt.subplots()
ax1.set_title('n(' + r'$\lambda$' + ')')
ax1.plot(wavelength, refractive_index('o', wavelength/1e3, 0), label='ord.')
ax1.plot(wavelength, refractive_index('e', wavelength/1e3, np.pi/2), label='ext.')
ax1.plot(wavelength, refractive_index('e', wavelength/1e3, angle), label='ext. ' + r'$\theta$=' + str(np.round(angle, 2)))
ax1.legend()
ax1.set_xlabel('$\lambda$ (nm)')
ax1.set_ylabel('n')


#%% setup SPDC experiment - type I matching signal(ord) + idler(ord) = pump(ext)
wavelength = np.linspace(300, 700) # unit: nm
angle = np.pi/4 # optic axis (OA) angle, unit: rad
D = 1 # distance between crystal and detector, unit: m

_, ax2 = plt.subplots()
ax2.set_title('output ring at varying ' + r'$\lambda$')
t = np.linspace(0, 2*np.pi, 100)
theta_c = np.zeros_like(wavelength)
for i in range(wavelength.shape[0]):
    theta_c[i] = emission_angle(wavelength[i], angle)
    ring = np.array([D*theta_c[i]*np.cos(t) , D*theta_c[i]*np.sin(t)])
    ax2.plot(ring[0,:], ring[1,:])
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')

_, ax3 = plt.subplots()
ax3.set_title('emission angle')
ax3.plot(2*wavelength,  theta_c, label='photon 1')
ax3.plot(2*wavelength, -theta_c, label='photon 2')
ax3.legend()
ax3.set_xlabel('output $\lambda$ (nm)')
ax3.set_ylabel('output ' + r'$\theta$ (rad)')


#%% polarization entangled source
wavelength = 400 # unit: nm
angle = np.pi/4 # optic axis (OA) angle, unit: rad
N = 10

_, ax4 = plt.subplots(1,2)
ax4[0].set_title('camera H')
ax4[1].set_title('camera V')
for i in range(N):
    q = entangled_photons(wavelength, angle)
    x1, y1 = D*q.theta*np.cos(q.phi), D*q.theta*np.sin(q.phi)
    x2, y2 =-D*q.theta*np.cos(q.phi),-D*q.theta*np.sin(q.phi)
    if q.polarization == 0:
        ax4[0].scatter([x1, x2], [y1, y2])
    else:
        ax4[1].scatter([x1, x2], [y1, y2])
ax4[0].set_xlabel('x (m)')
ax4[0].set_xlabel('y (m)')
ax4[1].set_xlabel('x (m)')
ax4[1].set_xlabel('y (m)')
ax4[0].set_aspect('equal', 'box')
ax4[1].set_aspect('equal', 'box')

i = 1