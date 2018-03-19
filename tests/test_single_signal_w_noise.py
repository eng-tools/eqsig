__author__ = 'maximmillen'

import numpy as np

from eqsig.single import Signal, AccSignal
from eqsig import checking_tools as ct
from tests.conftest import TEST_DATA_DIR


def load_test_record_from_file(record_path, record_filename, scale=1):
    a = open(record_path + record_filename, 'r')
    b = a.readlines()
    a.close()

    acc = []
    motion_step = float(b[0].split("=")[1])
    print('values dt: ', motion_step)
    for i in range(len(b)):
        if i > 3:
            dat = b[i].split()
            for j in range(len(dat)):
                acc.append(float(dat[j]) * scale)

    rec = AccSignal(acc, motion_step)
    return rec


def test_remove_polyfit_1():
    record_path = TEST_DATA_DIR
    record_filename = 'noise_test_1.txt'
    rec = load_test_record_from_file(record_path, record_filename)
    rec.remove_poly(poly_fit=1)
    ssq_org = np.sum(rec.values ** 2)

    rec.add_series(np.linspace(0, 0.2, rec.npts))
    ssq_w_linear = np.sum(rec.values ** 2)

    rec.remove_poly(poly_fit=1)
    ssq_corrected = np.sum(rec.values ** 2)

    assert ssq_org != ssq_w_linear
    assert ct.isclose(ssq_org, ssq_corrected)


def test_remove_polyfit_2():
    record_path = TEST_DATA_DIR
    record_filename = 'noise_test_1.txt'
    rec = load_test_record_from_file(record_path, record_filename)
    rec.remove_poly(poly_fit=2)
    ssq_org = np.sum(rec.values ** 2)

    x = np.linspace(0, 1.0, rec.npts)
    rec.add_series(0.2 * x - 0.5 * x ** 2)
    ssq_w_linear = np.sum(rec.values ** 2)

    rec.remove_poly(poly_fit=2)
    ssq_corrected = np.sum(rec.values ** 2)

    assert ssq_org != ssq_w_linear
    assert ct.isclose(ssq_org, ssq_corrected)


def test_butterpass():
    record_path = TEST_DATA_DIR
    record_filename = 'noise_test_1.txt'
    rec = load_test_record_from_file(record_path, record_filename)
    ssq_org = np.sum(rec.values ** 2)

    x = np.linspace(0, 1.0, rec.npts)
    rec.add_series(0.2 * x - 0.5 * x ** 2)
    ssq_w_linear = np.sum(rec.values ** 2)

    rec.butter_pass([0.2, 25])
    ssq_corrected = np.sum(rec.values ** 2)

    assert ssq_org != ssq_w_linear
    assert ct.isclose(ssq_org, 31854.72888, rel_tol=0.0001)
    assert ct.isclose(ssq_corrected, 10663.4862479, rel_tol=0.0001)


def test_fourier_spectra():
    recLocpath = TEST_DATA_DIR
    record = 'noise_test_1.txt'

    # Using eqsig
    cr = load_test_record_from_file(recLocpath, record)
    fa = cr.fa_spectrum
    ff = cr.fa_frequencies

    # bf, sf = plt.subplots(nrows=4, ncols=1, sharex=True)
    #
    # # sf[0].plot(time, acc)
    # sf[1].plot(ff, fa, label='eqsig')
    #
    # sf[1].legend(loc='upper right', prop={'size': 8})
    # sf[2].legend(loc='lower right', prop={'size': 8})
    # # bf.tight_layout()
    # plt.show()


def test_duration_stats():
    record_path = TEST_DATA_DIR
    record_filename = 'noise_test_1.txt'

    rec = load_test_record_from_file(record_path, record_filename, scale=9.8)
    rec.generate_duration_stats()
    assert ct.isclose(rec.t_595, 18.03)


