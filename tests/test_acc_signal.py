import numpy as np
from eqsig import loader
from tests import conftest
from tests.conftest import TEST_DATA_DIR
import eqsig


def test_arias_intensity(asig_t1):
    true_arias_intensity = 0.63398
    assert np.isclose(eqsig.im.calc_arias_intensity(asig_t1)[-1], true_arias_intensity, rtol=0.0001)


def test_cumulative_absolute_velocity(asig_t1):
    true_cav = 8.53872
    assert np.isclose(eqsig.im.calc_cav(asig_t1)[-1], true_cav, rtol=0.0001)


def test_peak_values(asig_t1):
    true_pga = 1.41
    true_pgv = 0.26006
    true_pgd = 0.07278134  # eqsig==0.4.12
    assert np.isclose(asig_t1.pga, true_pga, rtol=0.001)
    assert np.isclose(asig_t1.pgv, true_pgv, rtol=0.0001), asig_t1.pgv
    assert np.isclose(asig_t1.pgd, true_pgd, rtol=0.0001), asig_t1.pgd


def test_displacement_velocity(asig_t1):
    record_path = TEST_DATA_DIR
    # Compare time series
    test_filename = 'test_motion_avd.csv'
    data = np.loadtxt(record_path + test_filename, skiprows=1, delimiter=",")
    time = data[:, 0]
    velocity = data[:, 2]
    displacement = data[:, 3]
    assert len(time) == len(asig_t1.time)
    abs_velocity_diff = abs(asig_t1.velocity - velocity)
    cum_velocity_diff = sum(abs_velocity_diff)
    max_velocity_diff = max(abs_velocity_diff)
    assert cum_velocity_diff < 0.03, cum_velocity_diff
    assert max_velocity_diff < 0.00006, max_velocity_diff
    abs_disp_diff = abs(asig_t1.displacement - displacement)
    cum_disp_diff = sum(abs_disp_diff)
    max_disp_diff = max(abs_disp_diff)
    assert cum_disp_diff < 0.02, cum_disp_diff
    assert max_disp_diff < 0.00002, max_disp_diff

    # Compare time series versus true
    test_filename = 'test_motion_avd.csv'
    data = np.loadtxt(record_path + test_filename, skiprows=1, delimiter=",")
    time = data[:, 0]
    velocity = data[:, 2]
    displacement = data[:, 3]
    assert len(time) == len(asig_t1.time)
    abs_velocity_diff = abs(asig_t1.velocity - velocity)
    cum_velocity_diff = sum(abs_velocity_diff)
    max_velocity_diff = max(abs_velocity_diff)
    assert cum_velocity_diff < 0.03, cum_velocity_diff
    assert max_velocity_diff < 0.00006, max_velocity_diff
    abs_disp_diff = abs(asig_t1.displacement - displacement)
    cum_disp_diff = sum(abs_disp_diff)
    max_disp_diff = max(abs_disp_diff)
    assert cum_disp_diff < 0.02, cum_disp_diff
    assert max_disp_diff < 0.00002, max_disp_diff


def rewrite_response_spectra_eqsig_test_file(asig_t1):
    record_path = TEST_DATA_DIR

    s_a = asig_t1.s_a
    s_d = asig_t1.s_d
    times = asig_t1.response_times

    paras = []
    for i in range(len(times)):
        paras.append("%.5f,%.5f,%.5f" % (times[i], s_a[i], s_d[i]))
    outfile_name = record_path + "test_motion_dt0p01_rs.txt"
    outfile = open(outfile_name, "w")
    outfile.write("\n".join(paras))
    outfile.close()


def test_response_spectra_versus_old_eqsig_version(asig_t1):
    record_path = TEST_DATA_DIR
    s_a = asig_t1.s_a
    s_d = asig_t1.s_d
    times = asig_t1.response_times

    paras = []
    for i in range(len(times)):
        paras.append("%.5f,%.5f,%.5f" % (times[i], s_a[i], s_d[i]))

    testfile_name = record_path + "test_motion_dt0p01_rs.txt"
    testfile = open(testfile_name, "r")
    test_lines = testfile.readlines()
    for i, line in enumerate(test_lines):
        line = line.replace("\n", "")
        assert line == paras[i], i


def test_response_spectra(asig_t1):
    record_path = TEST_DATA_DIR
    test_filename = 'test_motion_true_spectra_acc.csv'
    data = np.loadtxt(record_path + test_filename, skiprows=1, delimiter=",")
    times = data[40:, 0]
    ss_s_a = data[40:, 1]

    asig_t1.response_times = times
    s_a = asig_t1.s_a
    s_a_in_g = s_a / 9.81
    a_times = asig_t1.response_times
    assert len(times) == len(a_times)
    srss1 = sum(abs(s_a_in_g - ss_s_a))
    assert srss1 < 0.105, srss1  # TODO: improve this at low frequencies!!!


def test_response_spectra_at_high_frequencies(asig_t1):
    record_path = TEST_DATA_DIR
    test_filename = 'test_motion_true_spectra_acc.csv'
    data = np.loadtxt(record_path + test_filename, skiprows=1, delimiter=",")
    times = data[:40, 0]
    ss_s_a = data[:40, 1]

    asig_t1.response_times = times
    s_a = asig_t1.s_a
    s_a_in_g = s_a / 9.81
    a_times = asig_t1.response_times
    assert len(times) == len(a_times)
    srss1 = sum(abs(s_a_in_g - ss_s_a))
    assert srss1 < 0.01 * 40, srss1


def test_duration_stats(asig_t1):
    asig_t1.generate_duration_stats()

    assert np.isclose(eqsig.im.calc_sig_dur(asig_t1), 20.99)  # eqsig==0.5.0
    assert np.isclose(eqsig.im.calc_brac_dur(asig_t1, 0.01), 38.27)  # eqsig==0.5.0
    assert np.isclose(eqsig.im.calc_brac_dur(asig_t1, 0.05), 15.41)  # eqsig==0.5.0
    assert np.isclose(eqsig.im.calc_brac_dur(asig_t1, 0.1), 8.41)  # eqsig==0.5.0

# 
# if __name__ == '__main__':
#     show_response_spectra_at_high_frequencies()
