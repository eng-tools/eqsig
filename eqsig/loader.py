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


def load_asig(ffp):
    vals, dt = load_values_and_dt(ffp)
    return AccSignal(vals, dt)


def save_signal(ffp, signal):
    save_values_and_dt(ffp, signal.values, signal.dt, signal.label)

