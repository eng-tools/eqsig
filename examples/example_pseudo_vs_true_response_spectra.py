import numpy as np
import eqsig
import matplotlib.pyplot as plt
from tests.conftest import TEST_DATA_DIR


# Array based input example

# The majority of eqsig's functions have been implemented using numpy arrays, while it is recommended to 
# use AccSignal object as an input, you can also use the array-based version of the function.


acc_record_filename = 'test_motion_dt0p01.txt'
acc = np.loadtxt(acc_record_filename, skiprows=2)
dt = 0.01

periods = np.linspace(0.01, 5, 40)
xi = 0.05
spectral_disp, pspectral_velo, pspectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=xi)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.true_response_spectra(acc, dt, periods, xi=xi)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))

sps[0].plot(periods, pspectral_acc / 9.8, c='b')
sps[1].plot(periods, pspectral_velo, c='b', label='Pseudo')
sps[2].plot(periods, spectral_disp, c='b')

sps[0].plot(periods, spectral_acc / 9.8, c='r', ls='--', label='Peak')
sps[1].plot(periods, spectral_velo, c='r', ls='--')

sps[1].plot(periods, spectral_acc / (2 * np.pi / periods), c='g', ls='--', label='a*w')


sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
sps[1].legend()

plt.show()