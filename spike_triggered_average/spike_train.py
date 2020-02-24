# generating poisson spike trains under the assuption that dt is sufficiently small

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calc_rt_dt( r_t, tau_ref = 0.01, dt=0.001, r0 = 100):
       return r0 + (r_t - r0)*np.exp(-1*dt/tau_ref)


r0 = 100 # hz
dt = 0.001 # time interval
t = np.arange(0,10,dt)
x = np.random.uniform(low=0.0, high=1.0, size=len(t)) # generate random numbers sampled from Uniform distr

# constant firing rate
spike_idx_constant = np.argwhere(x < r0*dt).reshape((1,-1))
inter_spike_distance_const = np.diff(spike_idx_constant)
mean, std = np.mean(inter_spike_distance_const), np.std(inter_spike_distance_const)
coeff_of_var_const = mean/std * 100

# non-constant firing rate
spike_idx_refactor = [] # take refactoring into account, i.e. non-constant firing rate
r_t = r0
for i,_t in enumerate(t):
       if x[i] < r_t*dt:
              spike_idx_refactor.append(i)
              r_t = 0
       else:
              r_t = calc_rt_dt(r_t = r_t)

inter_spike_distance_refactor = np.diff(spike_idx_refactor)
mean, std = np.mean(inter_spike_distance_refactor), np.std(inter_spike_distance_refactor)
coeff_of_var_refactor = mean/std * 100


fig, ax = plt.subplots(1,1)
plt.sca(ax)
ax.set(title='Possion Spike Train Interval Histogram',
       xlabel='Interval (ms)', ylabel='frequency')
sns.distplot(inter_spike_distance_const)
sns.distplot(inter_spike_distance_refactor)
ax.legend(['No Refactory Period. CofV: {}'.format(np.round(coeff_of_var_const,2)),
           'With Refactory Period. CofV: {}'.format(np.round(coeff_of_var_refactor,2))])
fig.show()
fig.savefig('p_spike_train.png', format='png')
plt.close('all')





