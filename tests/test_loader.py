from eqsig import loader
from tests.conftest import TEST_DATA_DIR


def test_load():
    values, dt = loader.load_values_and_dt(TEST_DATA_DIR + "test_motion_dt0p01.txt")
    assert dt == 0.01


def test_load_signal():
    asig = loader.load_signal(TEST_DATA_DIR + "test_motion_dt0p01.txt", astype='acc_sig')
    assert asig.dt == 0.01


def test_load_asignal():
    asig = loader.load_asig(TEST_DATA_DIR + "test_motion_dt0p01.txt", load_label=True)
    assert asig.dt == 0.01


if __name__ == '__main__':
    test_load_asignal()
