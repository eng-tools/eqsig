import numpy as np
import matplotlib.pyplot as plt

from eqsig.single import Signal, AccSignal
from eqsig import checking_tools as ct
from tests.conftest import TEST_DATA_DIR


def show_test_motion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    acc_signal = AccSignal(rec, motion_step)
    acc_signal.generate_displacement_and_velocity_series()
    bf, sp = plt.subplots(3)
    sp[0].plot(acc_signal.time, acc_signal.values)
    sp[1].plot(acc_signal.time, acc_signal.velocity)
    sp[2].plot(acc_signal.time, acc_signal.displacement)
    plt.show()


def test_remove_polyfit_1():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    acc_signal = Signal(rec, motion_step)

    # Remove any trend
    acc_signal.remove_poly(poly_fit=1)
    ssq_cleaned = np.sum(acc_signal.values ** 2)

    # Add a trend
    acc_signal.add_series(np.linspace(0, 0.2, acc_signal.npts))
    ssq_w_linear = np.sum(acc_signal.values ** 2)

    # remove the trend
    acc_signal.remove_poly(poly_fit=1)
    ssq_corrected = np.sum(acc_signal.values ** 2)

    assert ssq_cleaned != ssq_w_linear
    assert ssq_cleaned == ssq_corrected


def test_butterpass():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    acc_signal = Signal(rec, motion_step)

    ssq_org = np.sum(acc_signal.values ** 2)

    x = np.linspace(0, 1.0, acc_signal.npts)
    acc_signal.add_series(0.2 * x - 0.5 * x ** 2)
    ssq_w_linear = np.sum(acc_signal.values ** 2)

    acc_signal.butter_pass([0.2, 25])
    ssq_corrected = np.sum(acc_signal.values ** 2)

    assert ssq_org != ssq_w_linear
    assert ct.isclose(ssq_org, 395.9361125, rel_tol=0.0001)
    assert ct.isclose(ssq_corrected, 393.7198723, rel_tol=0.0001)


def rewrite_fourier_spectra_test_file():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    acc_signal = AccSignal(rec, motion_step)
    fa_amplitudes = abs(acc_signal.fa_spectrum)
    fa_phases = np.angle(acc_signal.fa_spectrum)

    paras = []
    for i in range(len(acc_signal.fa_frequencies)):
        paras.append("%.5f,%.5f,%.5f" % (acc_signal.fa_frequencies[i], fa_amplitudes[i], fa_phases[i]))
    outfile_name = record_path + "test_motion_dt0p01_fas.txt"
    outfile = open(outfile_name, "w")
    outfile.write("\n".join(paras))
    outfile.close()


def test_fourier_spectra():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    acc_signal = AccSignal(rec, motion_step)
    fa_amplitudes = abs(acc_signal.fa_spectrum)
    fa_phases = np.angle(acc_signal.fa_spectrum)

    paras = []
    for i in range(len(acc_signal.fa_frequencies)):
        paras.append("%.5f,%.5f,%.5f" % (acc_signal.fa_frequencies[i], fa_amplitudes[i], fa_phases[i]))

    testfile_name = record_path + "test_motion_dt0p01_fas.txt"
    testfile = open(testfile_name, "r")
    test_lines = testfile.readlines()
    for i, line in enumerate(test_lines):
        line = line.replace("\n", "")
        assert line == paras[i], i






if __name__ == '__main__':
    # rewrite_fourier_spectra_test_file()
    # rewrite_response_spectra_test_file()
    # test_response_spectra()
    pass
    # test_fourier_spectra()
    # test_duration_stats()
    # to_be_tested_spectra()
