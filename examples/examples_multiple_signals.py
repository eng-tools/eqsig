import numpy as np
import matplotlib.pyplot as plt
from eqsig import multiple
from eqsig import Signal


def show_same_start():
    time = np.linspace(0, 102, 10200)
    acc = np.sin(time)
    dt = 0.01

    cluster = multiple.Cluster([acc, acc + 0.3], dt=dt)
    plt.plot(cluster.time, cluster.values_by_index(0), label="sig0")
    plt.plot(cluster.time, cluster.values_by_index(1), label="sig1")
    cluster.same_start()
    plt.plot(cluster.time, cluster.values_by_index(0), label="sig0 - same start")
    plt.plot(cluster.time, cluster.values_by_index(1), label="sig1 - same start")
    plt.legend()
    plt.show()


def show_time_match():
    time = np.linspace(0, 102, 1020)
    acc = np.sin(time)
    dt = 0.01

    cluster = multiple.Cluster([acc[:-6], acc[6:]], dt=dt)
    plt.plot(cluster.time, cluster.values_by_index(0), label="sig0")
    plt.plot(cluster.time, cluster.values_by_index(1), label="sig1")
    cluster.time_match(verbose=1)
    plt.plot(cluster.time, cluster.values_by_index(0), label="sig0 - same start")
    plt.plot(cluster.time, cluster.values_by_index(1), label="sig1 - same start")
    plt.legend()
    plt.show()


def run_combine():
    import matplotlib.pyplot as plt
    time = np.linspace(0, 100, 1000)
    w = 0.2
    amp = 1.0
    x = amp * np.sin(w * time)
    y = amp / 2 * np.cos(w * time)

    off_rad = np.radians(45.)
    adj = x * np.cos(off_rad) + y * np.sin(off_rad)
    adj_alt = y * np.cos(off_rad) - x * np.sin(off_rad)
    off_rad = np.radians(90.)
    adj2 = x * np.cos(off_rad) + y * np.sin(off_rad)
    plt.plot(x, y)
    plt.plot(adj, adj_alt, ls="--")
    plt.show()
    plt.plot(time, x)
    plt.plot(time, y)
    plt.plot(time, adj)
    plt.plot(time, adj2, ls="--")
    plt.show()


if __name__ == '__main__':
    run_combine()
