import numpy as np

from eqsig.single import AccSignal
from tests.conftest import TEST_DATA_DIR
from eqsig import stockwell
from eqsig.functions import interp_to_approx_dt


def test_stockwell_transform_then_inverse():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    acc_signal = AccSignal(rec, motion_step)
    acc2_signal = interp_to_approx_dt(acc_signal, 0.1)
    acc2_signal.swtf = stockwell.transform(acc2_signal.values)
    inv_signal = AccSignal(stockwell.itransform(acc2_signal.swtf), acc2_signal.dt)

    abs_error = abs(acc2_signal.values - inv_signal.values)
    assert np.sum(abs_error) / acc2_signal.npts < 0.001, np.sum(abs_error)
