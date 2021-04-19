import numpy as np
import matplotlib.pyplot as plt
import eqsig

from matplotlib import rc
rc('font', family='Helvetica', size=9, weight='light')
plt.rcParams['pdf.fonttype'] = 42


dt = 0.01
time = np.arange(0, 10, dt)
f1 = 0.5
factor = 10.
f2 = f1 * factor
acc = np.cos(2 * np.pi * time * f1) + factor / 5 * np.cos(2 * np.pi * time * f2)

asig = eqsig.AccSignal(acc, dt)

asig.swtf = eqsig.stockwell.transform(asig.values)

bf, ax = plt.subplots(nrows=2, sharex=True, figsize=(5.0, 4.0))

ax[0].plot(asig.time, asig.values, lw=0.7, c='b', label='Signal')

in_pcm = eqsig.stockwell.plot_stock(ax[1], asig)
ax[1].set_ylim([0.0, 10])
ax[0].set_xlim([0, 10])

ax[0].set_ylabel('Amplitude [$m/s^2$]', fontsize=8)
ax[1].set_ylabel('$\it{Stockwell}$\nFrequency [Hz]', fontsize=8)
ax[-1].set_xlabel('Time [s]', fontsize=8)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbaxes = inset_axes(ax[1], width="20%", height="3%", loc='upper right')
cbaxes.set_facecolor([1, 1, 1])
cb = plt.colorbar(in_pcm, cax=cbaxes, orientation='horizontal')
cb.outline.set_edgecolor('white')
cbaxes.tick_params(axis='both', colors='white')

ax[0].legend(loc='upper right')
for sp in ax:
    sp.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()

bf.savefig('stockwell-example.png', dpi=90)
plt.show()


