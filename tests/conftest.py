import os
import sys
from eqsig import loader
import pytest

import numpy as np

from eqsig import AccSignal

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
def asig_t1():
    return loader.load_signal(TEST_DATA_DIR + "test_motion_dt0p01.txt", astype='acc_sig')
