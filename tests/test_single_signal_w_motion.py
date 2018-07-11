import numpy as np

from eqsig.single import Signal, AccSignal
from eqsig import checking_tools as ct
from tests.conftest import TEST_DATA_DIR


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

    assert not ct.isclose(ssq_cleaned, ssq_w_linear)
    assert ct.isclose(ssq_cleaned, ssq_corrected)


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


def test_fourier_spectra_with_motion():
    record_path = TEST_DATA_DIR

    record_filename = 'test_motion_dt0p01.txt'
    motion_dt = 0.01
    rec = np.loadtxt(record_path + record_filename)

    rec2 = np.zeros(2 ** 13)
    rec2[:len(rec)] = rec
    acc_signal = AccSignal(-rec, motion_dt)

    nfreq = len(acc_signal.fa_spectrum)
    test_filename = 'test_motion_true_fourier_spectra.csv'
    data = np.loadtxt(record_path + test_filename, skiprows=1, delimiter=",")
    freqs = data[:nfreq - 1, 0]
    fa = data[:nfreq - 1, 1]
    phase = data[:nfreq - 1, 2]

    fa_eqsig = abs(acc_signal.fa_spectrum)
    freq_eqsig = acc_signal.fa_frequencies
    org_phases = np.angle(acc_signal.fa_spectrum)
    ss_phases = np.angle(np.fft.rfft(rec2))[:len(org_phases)] + 0.0001

    for i in range(10):
        print(phase[i], acc_signal.fa_spectrum[i + 1], org_phases[i + 1])

    assert ct.isclose(freqs[0], freq_eqsig[1], rel_tol=0.001), freqs[0]
    assert ct.isclose(freqs[20], freq_eqsig[21], rel_tol=0.0001)
    assert ct.isclose(freqs[-1], freq_eqsig[-1], rel_tol=0.001)
    for i in range(len(fa)):
        assert ct.isclose(fa[i], fa_eqsig[i + 1], abs_tol=0.00001), i

    # bf, sp = plt.subplots(2)
    # sp[0].plot(freqs, fa, lw=0.5)
    # sp[0].plot(freq_eqsig, abs(acc_signal.fa_spectrum), lw=0.5)
    # sp[1].plot(freqs, phase, lw=0.5)
    # sp[1].plot(freq_eqsig, org_phases, lw=0.5)
    # sp[1].plot(freq_eqsig, ss_phases, lw=0.5)
    # plt.show()


# def test_fourier_with_sine_wave():
#     time = np.linspace(0, 10, 1000)
#     dt = time[1]
#     rec = np.sin(2 * np.pi * time)
#     acc = AccSignal(rec, dt)
#     bf, sp = plt.subplots(2)
#     sp[0].plot(time, rec)
#     sp[1].plot(acc.fa_frequencies, np.angle(acc.fa_spectrum))
#     plt.show()


def test_fourier_spectra_stable_against_aliasing():
    record_path = TEST_DATA_DIR

    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    rec2 = np.zeros(2 ** 13)
    rec2[:len(rec)] = rec
    org_signal = AccSignal(rec, motion_step)
    extended_signal = AccSignal(rec2, motion_step)

    rec_split = []
    for i in range(int(len(rec2) / 2)):
        rec_split.append(rec2[i * 2])

    acc_split = AccSignal(rec_split, motion_step * 2)

    org_fa = abs(org_signal.fa_spectrum)
    split_fa = abs(acc_split.fa_spectrum)
    ext_fa = abs(extended_signal.fa_spectrum)

    org_freq = abs(org_signal.fa_frequencies)
    split_freq = abs(acc_split.fa_frequencies)
    ext_freq = abs(extended_signal.fa_frequencies)

    for i in range(len(org_signal.fa_spectrum)):

        if i > 1830:
            abs_tol = 0.03
        else:
            abs_tol = 0.02

        assert ct.isclose(org_freq[i], ext_freq[i])
        assert ct.isclose(org_fa[i], ext_fa[i])

        if i < 2048:
            assert ct.isclose(org_freq[i], split_freq[i])
            assert ct.isclose(org_fa[i], split_fa[i], abs_tol=abs_tol), i


if __name__ == '__main__':
    # test_fourier_with_sine_wave()
    test_fourier_spectra_with_motion()

    # test_fourier_spectra_with_motion()
    # rewrite_fourier_spectra_test_file()
    # rewrite_response_spectra_test_file()
    # test_response_spectra()
    pass
    # test_fourier_spectra()
    # test_duration_stats()
    # to_be_tested_spectra()
