import numpy as np
import matplotlib.pyplot as plt
from eqsig import multiple
from eqsig import Signal


def failing_same_start():
    time = np.linspace(0, 1020, 10200)
    acc = np.sin(time)
    dt = 0.01
    sig0 = Signal(acc[:-20], dt)
    sig1 = Signal(acc[20:], dt)
    cluster = multiple.Cluster([sig0, sig1], dt=0.1)
    plt.plot(cluster.time, cluster.signal_by_index(0), label="sig0")
    plt.plot(cluster.time, cluster.signal_by_index(1), label="sig1")
    cluster.same_start()
    plt.plot(cluster.time, cluster.signal_by_index(0), label="sig0 - same start")
    plt.plot(cluster.time, cluster.signal_by_index(1), label="sig1 - same start")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_same_start()