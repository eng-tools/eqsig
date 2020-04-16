import numpy as np
import eqsig
import matplotlib.pyplot as plt
from tests import conftest
#%%
# acc = np.loadtxt('test_motion_dt0p01.txt', skiprows=2)
acc = np.loadtxt(conftest.TEST_DATA_DIR + 'short_motion_dt0p01.txt', skiprows=2)
dt = 0.01

from scipy import fft
npts = len(acc)
# if n_pad:
#     n_factor = 2 ** int(np.ceil(np.log2(npts)))
#     fa = fft(sig.values, n=n_factor)
#     points = int(n_factor / 2)
#     assert len(fa) == n_factor
# else:
fa = fft(acc)
points = int(npts / 2)
fas = fa[range(points)] * dt

n = 2 * len(fas)
a = np.zeros(2 * len(fas), dtype=complex)
a[1:n // 2] = fas[1:]
a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
a /= dt
s = np.fft.ifft(a)
npts = int(2 ** (np.log(n) / np.log(2)))
s = s[:npts]


st = eqsig.stockwell.transform(acc)

# bf, sps = plt.subplots()
# eqsig.stockwell.plot_tifq_vals(sps, abs(st), dt)
# asig = eqsig.AccSignal(acc, dt)
# eqsig.stockwell.plot_stock(sps, asig)
# plt.show()

#%%
# acc_new = eqsig.stockwell.itransform(st)
from scipy.fftpack import ifft  # Try use scipy.fft
ss = np.sum(st, axis=1)
n = 2 * len(ss)
fas_ss = np.zeros(2 * len(ss), dtype=complex)
fas_ss[1:n // 2] = np.flip(np.conj(ss[1:]), axis=0)
fas_ss[n // 2 + 1:] = ss[1:]
# fas_ss /= dt
plt.plot(a, c='k')
plt.plot(fas_ss, c='r')
plt.show()

acc_new = np.fft.ifft(fas_ss)
npts = int(2 ** (np.log(n) / np.log(2)))
acc_new = acc_new[:npts]
# acc_new = np.real(ifft(fas_ss))[::-1]
plt.plot(dt * np.arange(len(acc)), acc, c='k')
plt.plot(dt * np.arange(len(acc_new)), acc_new, c='r')
plt.show()