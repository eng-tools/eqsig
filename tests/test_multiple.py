import numpy as np
from eqsig import multiple


def test_same_start():
    time = np.linspace(0, 102, 10200)
    acc = np.sin(time)
    dt = 0.01

    cluster = multiple.Cluster([acc, acc + 0.3], dt=dt)
    cluster.same_start()
    diff = np.sum(cluster.values_by_index(0) - cluster.values_by_index(1))
    assert diff < 1.0e-10, diff


def test_time_match():
    time = np.linspace(0, 102, 1020)
    acc = np.sin(time)
    dt = 0.01

    cluster = multiple.Cluster([acc[:-6], acc[6:]], dt=dt)

    cluster.time_match(verbose=0)
    diff = np.sum(cluster.values_by_index(0)[6:-5] - cluster.values_by_index(1)[6:-5])
    assert diff == 0.0, diff


if __name__ == '__main__':
    test_same_start()
    # test_time_match()
