import numpy as np

import eqsig


def test_roll_av_vals():
    expected = np.array([4, 4, 3, 2, 1, 1, 1, 1])
    assert np.sum(eqsig.fns.average.calc_roll_av_vals([4, 4, 4, 4, 1, 1, 1, 1], steps=3) - expected) == 0
    expected = np.array([4, 4, 4, 4, 3, 2, 1, 1])
    assert np.sum(
        eqsig.fns.average.calc_roll_av_vals([4, 4, 4, 4, 1, 1, 1, 1], steps=3, mode='backward') - expected) == 0
    expected = np.array([4, 4, 4, 3, 2, 1, 1, 1])
    assert np.sum(eqsig.fns.average.calc_roll_av_vals([4, 4, 4, 4, 1, 1, 1, 1], steps=3, mode='centre') - expected) == 0
