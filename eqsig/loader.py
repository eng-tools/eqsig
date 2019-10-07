import numpy as np
from eqsig.single import Signal, AccSignal


def load_values_and_dt(ffp):
    """
    Loads values and time step that were saved in eqsig input format.

    Parameters
    ----------
    ffp: str
        Full file path to output file

    Returns
    -------
    values: array_like
        An array of values
    dt: float
        Time step

        """
    data = np.genfromtxt(ffp, skip_header=1, delimiter=",", names=True, usecols=0)
    dt = data.dtype.names[0].split("_")[-1]
    dt = "." + dt[1:]
    dt = float(dt)
    values = data.astype(np.float)
    return values, dt


# def fast_nga_loader(ffp):
#     data = np.genfromtxt(ffp, skip_header=4, names=True)
#     data.flatten()
#     dt = data.dtype.names[0].split("DT=")[-1]
#     dt = "." + dt[1:]
#     print(dt)
#     dt = float(dt)
#     values = data.astype(np.float)
#     return values, dt


def save_values_and_dt(ffp, values, dt, label):
    """
    Exports acceleration values to the eqsig format.

    Parameters
    ----------
    ffp: str
        Full file path to output file
    values: array_like
        An array of values
    dt: float
        Time step
    label: str
        A label of the data

    Returns
    -------

    """
    para = [label, "%i %.4f" % (len(values), dt)]
    for i in range(len(values)):
        para.append("%.6f" % values[i])
    ofile = open(ffp, "w")
    ofile.write("\n".join(para))
    ofile.close()


def load_signal(ffp, astype='sig'):
    vals, dt = load_values_and_dt(ffp)
    if astype == "signal":
        return Signal(vals, dt)
    elif astype == "acc_sig":
        return AccSignal(vals, dt)


def load_sig(ffp):
    vals, dt = load_values_and_dt(ffp)
    return Signal(vals, dt)


def load_asig(ffp, load_label=False, m=1.0):
    vals, dt = load_values_and_dt(ffp)
    if load_label:
        a = open(ffp)
        label = a.read().splitlines()[0]
        a.close()
    else:
        label = 'm1'
    return AccSignal(vals * m, dt, label=label)


def save_signal(ffp, signal):
    save_values_and_dt(ffp, signal.values, signal.dt, signal.label)


def load_3_comp_values_and_dt_from_v2a(ffp):
    """
    Loads a ground motion file stored in the V2A format

    Parameters
    ----------
    ffp

    Returns
    -------

    """
    a = open(ffp)
    b = a.readlines()
    a.close()
    npts = None
    dt = None
    for line in b:
        if 'Number of points' in line:
            parts = line.split('points')[1]
            n_str = parts.split('Duration')[0]
            npts = int(n_str)
        if 'corrected data at' in line:
            parts = line.split('data at')[1]
            dt_str = parts.split('sec')[0]
            dt = float(dt_str)
        if npts is not None and dt is not None:
            break
    b = b[26:]
    elines = int(np.ceil(float(npts) / 10))
    accs = []
    for i in range(3):
        accs.append([])
        print(3 * i * elines, (3 * i + 1) * elines)
        for j in range(3 * i * elines + i * 26, (3 * i + 1) * elines + i * 26):
            accs[i] += b[j].split()
    accs = np.array(accs).astype(float) / 1e3

    return accs[0], accs[1], accs[2], dt

