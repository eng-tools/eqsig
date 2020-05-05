import eqsig
import matplotlib.pyplot as plt
import numpy as np
from tests import conftest


def create():
    asig = eqsig.load_asig(conftest.TEST_DATA_DIR + 'test_motion_dt0p01.txt')

    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(asig.time, asig.values, c='k', lw=1)
    asig.gen_fa_spectrum(p2_plus=0)
    sps[1].loglog(asig.fa_freqs, abs(asig.fa_spectrum), c='b', label='FAS')
    sps[1].loglog(asig.smooth_fa_freqs, asig.smooth_fa_spectrum, c='r', label='band=40 (default)')
    asig.gen_smooth_fa_spectrum(band=100)
    sps[1].loglog(asig.smooth_fa_freqs, asig.smooth_fa_spectrum, c='orange', label='band=100')
    asig.gen_smooth_fa_spectrum(band=40, smooth_fa_freqs=asig.fa_freqs[1:])
    sps[1].loglog(asig.smooth_fa_freqs, asig.smooth_fa_spectrum, c='y', label='band=40 - full width')

    # Using functions instead of inbuilt methods
    smatrix = eqsig.calc_smoothing_matrix_konno_1998(asig.fa_freqs, band=40)
    sps[1].loglog(asig.fa_freqs[1:], eqsig.calc_smooth_fa_spectrum_w_custom_matrix(asig, smatrix), c='g', label='band=40 (full w fns)', ls='--')

    narrow_freqs = np.logspace(-1, 1, 100, base=10)
    smatrix = eqsig.calc_smoothing_matrix_konno_1998(asig.fa_freqs, smooth_fa_frequencies=narrow_freqs, band=40)
    sps[1].loglog(narrow_freqs, eqsig.calc_smooth_fa_spectrum_w_custom_matrix(asig, smatrix), c='m',
                  label='band=40 (narrow w fns)', ls=':')
    plt.legend(prop={'size': 7})
    plt.show()


if __name__ == '__main__':
    create()
