# PERSISTENT SODIUM MODEL
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# declare system constants
C = 10e-3 # capacitance
I_e = 0 # no injected current
g_l = 10e-3 # conductances in Siemens
E_l = -67e-3
g_Na = 74e-3
E_Na = 60e-3 # electric field strength, Volts
V_half = 1.5e-3
k = 16e-3


# proceed to solve the differential equation
def dEm_dt(E_m, t, I_e = I_e):
    I_l = g_l*(E_m - E_l)

    x = (V_half - E_m) / k
    m_inf = 1/(1 + np.exp(x))
    I_Na_p = g_Na*m_inf*(E_m - E_Na)

    I_m = I_l + I_Na_p

    dEm_dt = 1/C * (I_e - I_m) # I_e - I_m
    return dEm_dt

# solve the ODE using ODEINT
t = np.arange(0,20,0.001)
E_m = np.array(odeint(dEm_dt,y0=-0.1, t=t))

# plot the solution E_m(t)
plt.plot(t, E_m)
plt.show()
plt.savefig('E_m.png', format='png')
plt.close(plt.gcf())

# plot the different currents as a function of E_m
I_l = g_l*(E_m - E_l)

x = (V_half - E_m) / k
m_inf = 1/(1 + np.exp(x))
I_Na_p = g_Na*m_inf*(E_m - E_Na)

I_m = I_l + I_Na_p

fig, ax = plt.subplots(1,1)
plt.sca(ax)
ax.plot(E_m, I_l, E_m, I_Na_p, E_m, I_m)
ax.legend(['I_l','I_Na_p','I_m'])
ax.set(xlabel='E_m (mV)', ylabel='I (mA)')
fig.show()
fig.savefig('currents.png', format='png')


# plot E_m and I_m as a function of time for different initial
# conditions, to demonstrate that a proper selection nof
# E_m(0) will yield a good approximation of the upstroke of the neuron
y0 = [-5, -1, -.8, -0.5, -0.1, 0]

fig, axes = plt.subplots(3,2)
for ax,_y0 in zip(axes.flatten(),y0):
    plt.sca(ax)
    E_m = np.array(odeint(dEm_dt, y0=_y0, t=t))
    ax.plot(t, E_m, t, I_m)
    ax.legend(['E_m', 'I_m'])
    ax.set(ylabel='E_m (mV)', xlabel='time (ms)',
           title='E_m(0) = {}'.format(_y0))

fig.show()
fig.savefig('em_im.png', format='png')
plt.close('all')





