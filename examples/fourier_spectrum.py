import eqsig
import matplotlib.pyplot as plt

from tests import conftest


def create():
    asig = eqsig.load_asig(conftest.TEST_DATA_DIR + 'test_motion_dt0p01.txt')
    asig_small = eqsig.interp_to_approx_dt(asig, 0.001)

    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(asig.time, asig.values)
    sps[0].plot(asig_small.time, asig_small.values, ls='--')
    sps[1].plot(asig.fa_frequencies, abs(asig.fa_spectrum))
    sps[1].loglog(asig_small.fa_frequencies, abs(asig_small.fa_spectrum), ls='--')

    plt.show()



if __name__ == '__main__':
    create()