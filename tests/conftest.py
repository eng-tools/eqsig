import os
import sys
from eqsig import loader, AccSignal
import pytest

import numpy as np


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(PACKAGE_DIR)
TEST_DIR = os.path.dirname(__file__)
OPENSEES_DATA_DIR = os.path.join(TEST_DIR, 'test_data/opensees_data/')
TEST_DATA_DIR = os.path.join(TEST_DIR, 'unit_test_data/')


def t_asig():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    return AccSignal(rec, motion_step)


@pytest.fixture()
def noise_rec():
    record_path = TEST_DATA_DIR
    record_filename = 'noise_test_1.txt'
    return load_test_record_from_file(record_path, record_filename)


@pytest.fixture()
def asig_t1():
    return loader.load_signal(TEST_DATA_DIR + "test_motion_dt0p01.txt", astype='acc_sig')


def load_test_record_from_file(record_path, record_filename, scale=1):
    a = open(record_path + record_filename, 'r')
    b = a.readlines()
    a.close()

    acc = []
    motion_step = float(b[0].split("=")[1])
    for i in range(len(b)):
        if i > 3:
            dat = b[i].split()
            for j in range(len(dat)):
                acc.append(float(dat[j]) * scale)

    rec = AccSignal(acc, motion_step)
    return rec
