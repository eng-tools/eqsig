import numpy as np

from eqsig import im

from tests import conftest


def test_cavdp():
    asig = conftest.t_asig()

    cav_dp = im.calc_cav_dp(asig)[-1]

    # 0.810412819 from eqsig 0.5.26  tested with several motions
    assert np.isclose(cav_dp, 0.81041282, rtol=0.001), cav_dp
