__author__ = 'maximmillen'

from collections import OrderedDict
# from sortedcontainers import SortedDict
import numpy as np

from eqsig.single import Signal, AccSignal


class Cluster(object):

    def __init__(self, signals, dt, names=[], master_index=0, stypes="custom", **kwargs):

        self.freq_range = np.array(kwargs.get('freq_range', [0.1, 20]))
        lvt = np.log10(1.0 / self.freq_range)
        if stypes == "custom" or stypes == "acc":
            stypes = [stypes] * len(signals)
        self.response_times = kwargs.get('resp_times', np.logspace(lvt[1], lvt[0], 31, base=10))
        if master_index < 0:
            raise ValueError("master_index must be positive")
        if master_index > len(signals) - 1:
            raise ValueError("master_index: %i, out of bounds, maximum value: %i" % (master_index, len(signals) - 1))
        self.master_index = master_index
        # if len(signals) != len(steps):
        #     raise ValueError("Length of signals: %i, must match length of steps: %i" % (len(signals), len(steps)))

        # self.master = Record(master_motion, master_step, response_times=self.response_times)
        shortage = len(signals) - len(names)
        self.names = list(names)
        for i in range(shortage):
            self.names.append("m%i" % (len(names) + i))
        self.master = self.names[master_index]
        self.dt = dt
        self.signals = OrderedDict()
        for s in range(len(signals)):
            if stypes[s] == "acc":
                self.signals[names[s]] = Signal(signals[s], dt, names[s], response_times=self.response_times)

    def values_by_index(self, index):
        key_value_pair = self.signals.items()[index]
        return key_value_pair[1].values

    def signal_by_index(self, index):
        key_value_pair = self.signals.items()[index]
        return key_value_pair[1]

    def values(self, name):
        return self.signals[name].values

    def name_by_index(self, index):
        key_value_pair = self.signals.items()[index]
        return key_value_pair[0]

    @property
    def time(self):
        sig0 = self.signal_by_index(0)
        return sig0.time

    def same_start(self, **kwargs):
        """
        Starts both time series at the same y-value
        """
        base = kwargs.get('base', 0)
        start = kwargs.get('start', 0)
        end = kwargs.get('end', 1)
        verbose = kwargs.get('verbose', 0)
        master_average = self.signal_by_index(self.master_index).get_section_average(start=start, end=end)
        for i in range(len(self.signals)):
            if i != self.master_index:
                slave_signal = self.signal_by_index(1)
                slave_average = slave_signal.get_section_average(start=start, end=end)
                diff = slave_average - master_average
                if verbose:
                    print('Same start the records')
                    print('old difference in starts: ', diff)
                slave_signal.values -= diff

    def combine_motions(self, f_ch, low_index=0, high_index=1, **kwargs):
        """
        This method combined the two ground motions by taking the high frequency of one values
        and the low frequency of the other.
        WARNING: records must have the same time step!!!
        f_ch: is the frequency change point
        order: refers to the order of the BW filter
        highfreq: if 0 then values[0] is used for high frequency content
                  if 1 then values[1] is used for high frequency content
        combined values is returned as self.combo
        """
        order = kwargs.get('order', 4)
        remove_gibbs = kwargs.get('remove_gibbs', 0)
        self.signal_by_index(high_index).butter_pass(cut_off=(f_ch, None), order=order, remove_gibbs=remove_gibbs)
        self.signal_by_index(low_index).butter_pass(cut_off=(None, f_ch), order=order, remove_gibbs=remove_gibbs)
        motion = self.signal_by_index(low_index).values + self.signal_by_index(high_index).values
        return motion

    def time_match(self, **kwargs):
        """
        This method determines the time lag between two records and removes the lag
        - the solution is based on a sum of the squared residuals.
        - The motions are both set to the base values, default = 0
        - must have same time step and same magnitude
        """
        verbose = kwargs.get('verbose', 0)
        steps = kwargs.get('steps', 10)
        set_step = kwargs.get('set_step', False)
        trim = kwargs.get('trim', True)

        if set_step is False:
            if verbose:
                print('length m0: ', self.signal_by_index(0).npts)
                print('length m1: ', self.signal_by_index(1).npts)
            length_check = min(self.signal_by_index(0).npts, self.signal_by_index(1).npts)
            bm = self.signal_by_index(self.master_index).motion[:length_check]
            for s in range(len(self.signals)):
                if s != self.master_index:
                    om = self.signal_by_index(s).motion[:length_check]
                    squares = (bm[0:-steps] - om[0:-steps]) ** 2
                    min_diff = np.sum(squares)
                    min_ind = 0
                    if verbose:
                            print('mi: ', min_ind, ' min_diff: ', min_diff)
                # Check other values lags base values
                for i in range(steps):
                    squares = (om[i:-steps + i] - bm[0:-steps]) ** 2
                    diff = sum(squares)
                    if verbose:
                        print('ind: ', i, ' diff: ', diff)
                    if diff < min_diff:
                        min_diff = diff
                        min_ind = i + 0
                # Check base values lags other values
                for i in range(steps):
                    squares = (bm[i:-steps + i] - om[0:-steps]) ** 2
                    diff = sum(squares)
                    if verbose:
                        print('ind: ', i, ' diff: ', diff)
                    if diff < min_diff:
                        min_diff = diff
                        min_ind = -i - 0

        # else:
        #     min_ind = set_step
        # if verbose:
        #     print('time step lag: ', min_ind)
        # if min_ind == 0:
        #     if verbose:
        #         print('no time lag')
        # elif min_ind < 0:
        #     m_temp = self.motions([np.mod(base + 1, 2)]).motion[:min_ind]
        # elif min_ind > 0:
        #     m_temp = self.motions([np.mod(base + 1, 2)]).motion[min_ind:]
        # # pad other record with constant
        # if verbose:
        #     print('m_temp[0]: ', m_temp[0])
        # if min_ind != 0:
        #     m = self.motions([np.mod(base + 1, 2)])
        #     m = np.ones(len(self.motions[np.mod(base + 1, 2)]))
        # if min_ind < 0:
        #     self.motions[np.mod(base + 1, 2)] = m_temp[0] * self.motions[np.mod(base + 1, 2)]
        #     self.motions[np.mod(base + 1, 2)][-min_ind:] = m_temp
        # elif min_ind > 0:
        #     self.motions[np.mod(base + 1, 2)] = m_temp[-1] * self.motions[np.mod(base + 1, 2)]
        #     self.motions[np.mod(base + 1, 2)][:-min_ind] = m_temp
        # if trim:
        #     tmax = len(self.motions[base]) * self.dts[base]
        #     ender = int(tmax / self.dts[np.mod(base + 1, 2)])
        #     if verbose:
        #         print('original length: ', len(self.motions[np.mod(base + 1, 2)]))
        #         print('trim length: ', ender)
        #     self.motions[np.mod(base + 1, 2)] = self.motions[np.mod(base + 1, 2)][:ender]
        return min_ind

    def generate_response_spectrums(self):
        for i in range(len(self.signals)):
            self.signal_by_index(i).generate_response_spectrum()

    def calculate_ratios(self):

        self.generate_response_spectrums()

        self.resp_log_ratio = np.log10(self.motions(1).s_a / self.motions(0).s_a)
        self.fa_log_ratio = np.log10(self.motions(1).smooth_fa_spectrum / self.motions(0).smooth_fa_spectrum)
        centre_freq = 1.0 / self.response_times[len(self.response_times) / 2]

        self.response_ratio_moment = sum(self.resp_log_ratio * (np.log10((1.0 /
                                         self.response_times) / centre_freq)))


def combine_at_angle(acc_sig_ns, acc_sig_we, angle):
    off_rad = np.radians(angle)
    combo = acc_sig_ns.values * np.cos(off_rad) + acc_sig_we.values * np.sin(off_rad)
    new_sig = AccSignal(combo, acc_sig_ns.dt)
    return new_sig


def compute_rotated(acc_sig_ns, acc_sig_we, angle_off_ns=0.0, parameter="arias_intensity", points=100):
    """
    Computes the rotated value of a parameter.
    :param acc_sig_ns:
    :param acc_sig_we:
    :param angle_off_ns: Angle from North in degrees of the primary signal.
    :return: tuple, (angle to rotate, rotated)
    """
    assert isinstance(acc_sig_ns, AccSignal)
    assert isinstance(acc_sig_we, AccSignal)
    assert acc_sig_ns.dt == acc_sig_we.dt
    assert acc_sig_ns.npts == acc_sig_we.npts, (acc_sig_ns.npts, acc_sig_we.npts)

    degrees = np.linspace(0 - angle_off_ns, 180. - angle_off_ns, points)
    degrees = np.mod(degrees, 360)
    pvalues = []
    for i in range(len(degrees)):
        new_sig = combine_at_angle(acc_sig_ns, acc_sig_we, degrees[i])
        new_sig.generate_all_motion_stats()
        pvalues.append(getattr(new_sig, parameter))

    return degrees, np.array(pvalues)


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