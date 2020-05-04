import numpy as np
import eqsig
import matplotlib.pyplot as plt
from tests import conftest
from scipy.fft import fft
#%%
# acc = np.loadtxt('test_motion_dt0p01.txt', skiprows=2)
# acc = np.loadtxt(conftest.TEST_DATA_DIR + 'short_motion_dt0p01.txt', skiprows=2)
# acc = np.loadtxt(conftest.TEST_DATA_DIR + 'test_motion_dt0p01.txt', skiprows=2)
acc = np.loadtxt(conftest.TEST_DATA_DIR + 'RSN5616-rot100AI.txt', skiprows=2)
print(conftest.TEST_DATA_DIR + 'RSN5616-rot100AI.txt')
acc = acc[:18199]
dt = 0.01


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
aa = np.zeros(2 * len(fas), dtype=complex)
aa[1:n // 2] = fas[1:]
aa[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
aa /= dt
s = np.fft.ifft(aa)
npts = int(2 ** (np.log(n) / np.log(2)))
acc_np = s[:npts]


# st = eqsig.stockwell.transform_w_scipy_fft(acc)
st = eqsig.stockwell.transform(acc)
acc_new_m = eqsig.stockwell.itransform(st)
time_off = 0
if time_off:
    ss = np.sum(st, axis=1)
    # ss[:1000] = 0
    # ss_dep = np.sum(st_dep, axis=1)
    # print(len(ss), len(ss_dep))
    n = 2 * len(ss)
    fas_ss = np.zeros(2 * len(ss), dtype=complex)
    fas_ss[1:n // 2] = np.flip(np.conj(ss[1:]), axis=0)
    fas_ss[n // 2 + 1:] = ss[1:]
    # fas_ss /= dt

    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(aa, c='b', lw=1)
    sps[0].plot(fas_ss, c='r', lw=1, ls='--')
    # sps[0].plot(aa, c='r')

    acc_new = np.fft.ifft(fas_ss)
    npts = int(2 ** (np.log(n) / np.log(2)))
    # acc_new = acc_new[:npts][::-1]

    sps[1].plot(dt * np.arange(len(acc)), acc, c='k')
    sps[1].plot(dt * np.arange(len(acc_new)), acc_new, c='r')
    # sps[1].plot(dt * np.arange(len(acc_np)), acc_np, c='b', lw=0.7, ls='-.')
    sps[1].plot(dt * np.arange(len(acc_new_m)), acc_new_m, c='g', ls='--')
    plt.show()