import numpy as np
import matplotlib.pyplot as plt

from eqsig.single import AccSignal
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


def show_response_spectra_at_high_frequencies():
    record_path = TEST_DATA_DIR
    test_filename = 'test_motion_true_spectra_acc.csv'
    data = np.loadtxt(record_path + test_filename, skiprows=1, delimiter=",")
    times = data[:40, 0]
    ss_s_a = data[:40, 1]

    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    # acc_signal = AccSignal(rec, motion_step, response_times=times)
    # s_a = acc_signal.s_a
    #
    # a_times = acc_signal.response_times
    # s_d, s_v, s_a = dh.pseudo_response_spectra(rec, motion_step, times, xi=0.05)
    # s_d, s_v, s_a = dh.true_response_spectra(rec, motion_step, times, xi=0.05)
    acc_signal = AccSignal(rec, motion_step, response_times=times)
    s_a = acc_signal.s_a

    s_a_in_g = s_a / 9.81

    # srss1 = sum(abs(s_a_in_g - ss_s_a))
    plt.plot(times, s_a_in_g, label="eqsig")
    plt.plot(times, ss_s_a, label="true-ish")
    plt.legend()
    plt.show()


def show_fourier_spectra_stable_against_aliasing():
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

    bf, sp = plt.subplots(2)
    sp[0].plot(org_signal.time, org_signal.values)
    sp[0].plot(extended_signal.time, extended_signal.values)
    sp[0].plot(acc_split.time, acc_split.values)

    sp[1].plot(org_signal.fa_frequencies, abs(org_signal.fa_spectrum), lw=0.7, label="original")
    sp[1].plot(acc_split.fa_frequencies, abs(acc_split.fa_spectrum), lw=0.7, label="split")
    sp[1].plot(extended_signal.fa_frequencies, abs(extended_signal.fa_spectrum), lw=0.7, label="full")


    plt.legend()
    plt.show()