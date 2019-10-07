from eqsig import sdof
import numpy as np
from tests.conftest import TEST_DATA_DIR


def test_nigam_and_jennings_vs_duhamels():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    dt = 0.01
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    periods = np.linspace(0.1, 2.5, 2)
    xi = 0.05
    resp_u, resp_v, sdof_acc = sdof.response_series(rec, dt, periods, xi)
    i = 0
    duhamel_u = sdof.single_elastic_response(rec, dt, periods[i], xi)
    diff_disp = np.sum(abs(resp_u[i] - duhamel_u))
    assert diff_disp < 2.0e-2, diff_disp
    i = 1
    duhamel_u = sdof.single_elastic_response(rec, dt, periods[i], xi)
    diff_disp = np.sum(abs(resp_u[i] - duhamel_u))
    assert diff_disp < 3.0e-2, diff_disp
    # import matplotlib.pyplot as plt
    # bf, sps = plt.subplots(figsize=(9, 5))
    # plt.plot(resp_u[i], lw=0.7)
    # plt.plot(duhamel_u,lw=0.7)
    # plt.show()


if __name__ == '__main__':
    test_nigam_and_jennings_vs_duhamels()

