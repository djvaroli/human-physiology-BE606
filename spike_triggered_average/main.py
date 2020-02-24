import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sample_rate = 500 # HZ, how often the samples were taken
dt = 1/sample_rate # time interval between readings
collection_time = 20*60 # seconds that experimented was carried out for
window = 150 # time steps, 300ms to calculate spike triggered average over
cols = ['Stimulus', 'Rho']
data = pd.read_csv('C1P8data.csv', names=cols)


spike_idx = data.index[data['Rho'] == 1].tolist() # indices of spikes
# print(spike_idx[0])
temp_matrix = [] # matrix to store stimulus for given window

stimulus = data['Stimulus'].to_numpy()

for i,idx in enumerate(spike_idx):
    # if i%100 == 0:
    #     print('{} / {}'.format(i + 1, len(spike_idx)))
    if idx - window > 0:
        current_window = stimulus[idx - window:idx].tolist()
        temp_matrix.append(current_window)


temp_matrix = np.array(temp_matrix) # convert to numpy
sta = np.mean(temp_matrix, axis=0) # calculate sp tr av

fig, ax = plt.subplots(1,1)
time = 2*np.arange(window) # convert time steps to ms
ax.plot(time, sta)
ax.set(xlabel='Time (ms)', ylabel='Stimulus', title='Spike Triggered Average')
fig.show()
fig.savefig('sptrav.png', format='png')
plt.close('all')
