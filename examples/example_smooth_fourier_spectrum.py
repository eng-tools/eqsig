import eqsig
import matplotlib.pyplot as plt

from tests import conftest


def create():
    asig = eqsig.load_asig(conftest.TEST_DATA_DIR + 'test_motion_dt0p01.txt')

    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(asig.time, asig.values, c='k', lw=1)
    sps[1].loglog(asig.fa_frequencies, abs(asig.fa_spectrum), c='b')
    sps[1].loglog(asig.smooth_fa_frequencies, asig.smooth_fa_spectrum, c='r')

    plt.show()


if __name__ == '__main__':
    create()
