__author__ = 'maximmillen'

from collections import OrderedDict
from sortedcontainers import SortedDict
import numpy as np

from single import Record

class Cluster(object):

    def __init__(self, master_motion, master_step, slave_motions=[], slave_steps=[], slave_names=[], **kwargs):

        self.freq_range = np.array(kwargs.get('freq_range', [0.1, 20]))
        lvt = np.log10(1.0 / self.freq_range)
        self.response_times = kwargs.get('resp_times', np.logspace(lvt[1], lvt[0], 31, base=10))

        self.master = Record(master_motion, master_step, response_times=self.response_times)
        if len(slave_motions) > slave_names:
            shortage = len(slave_motions) - slave_names
            for i in range(shortage):
                slave_names.append("m%i" % (len(slave_names) + i))
        self.slaves = SortedDict()
        for s in range(len(slave_motions)):
            self.slaves[slave_names[s]] = Record(slave_motions[s], slave_steps[s],
                                                 slave_names[s], response_times=self.response_times)



    def motions(self, index):
        if index == 0:
            return self.master
        return self.slaves.iloc[index - 1]

    def same_start(self, **kwargs):
        '''
        Starts both time series at the same y-value
        '''
        base = kwargs.get('base', 0)
        start = kwargs.get('start', 0)
        end = kwargs.get('end', 1)
        verbose = kwargs.get('verbose', 0)
        master_average = self.motions(0).get_section_average(start=start, end=end)
        slave_average = self.motions(1).get_section_average(start=start, end=end)
        diff = slave_average - master_average
        if verbose:
            print('Same start the records')
            print('old difference in starts: ', diff)

        slave = self.motions(1)
        slave.motion -= diff

    def combine_motions(self, f_ch, **kwargs):
        '''
        This method combined the two ground motions by taking the high frequency of one values
        and the low frequency of the other.
        WARNING: records must have the same time step!!!
        f_ch: is the frequency change point
        order: refers to the order of the BW filter
        highfreq: if 0 then values[0] is used for high frequency content
                  if 1 then values[1] is used for high frequency content
        combined values is returned as self.combo
        '''
        order = kwargs.get('order', 4)
        highfreq = kwargs.get('highfreq', 0)
        remove_Gibbs = kwargs.get('remove_Gibbs', 0)
        self.master.butter_pass(f_ch, single=highfreq, filter_type='highpass', order=order, remove_Gibbs=remove_Gibbs)
        self.motions(1).butter_pass(f_ch, single=np.mod(highfreq + 1, 2), filter_type='lowpass', order=order, remove_Gibbs=remove_Gibbs)
        motion = self.motions(0).motion + self.motions(1).motion
        return motion

    def time_match(self, **kwargs):
        '''
        This method determines the time lag between two records and removes the lag
        - the solution is based on a sum of the squared residuals.
        - The motions are both set to the base values, default = 0
        - must have same time step and same magnitude
        '''
        base = kwargs.get('base', 0)
        verbose = kwargs.get('verbose', 0)
        steps = kwargs.get('steps', 10)
        set_step = kwargs.get('set_step', False)
        trim = kwargs.get('trim', True)

        if set_step == False:
            if verbose:
                print('length m0: ', self.motions(0).npts)
                print('length m1: ', self.motions(1).npts)
            length_check = min(self.motions(0).npts, self.motions(1).npts)
            bm = self.motions([np.mod(base, 2)]).motion[:length_check]
            om = self.motions([np.mod(base + 1, 2)]).motion[:length_check]
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

        else:
            min_ind = set_step
        if verbose:
            print('time step lag: ', min_ind)
        if min_ind == 0:
            if verbose:
                print('no time lag')
        elif min_ind < 0:
            m_temp = self.motions([np.mod(base + 1, 2)]).motion[:min_ind]
        elif min_ind > 0:
            m_temp = self.motions([np.mod(base + 1, 2)]).motion[min_ind:]
        # pad other record with constant
        if verbose:
            print('m_temp[0]: ', m_temp[0])
        if min_ind != 0:
            m = self.motions([np.mod(base + 1, 2)])
            m = np.ones(len(self.motions[np.mod(base + 1, 2)]))
        if min_ind < 0:
            self.motions[np.mod(base + 1, 2)] = m_temp[0] * self.motions[np.mod(base + 1, 2)]
            self.motions[np.mod(base + 1, 2)][-min_ind:] = m_temp
        elif min_ind > 0:
            self.motions[np.mod(base + 1, 2)] = m_temp[-1] * self.motions[np.mod(base + 1, 2)]
            self.motions[np.mod(base + 1, 2)][:-min_ind] = m_temp
        if trim:
            tmax = len(self.motions[base]) * self.dts[base]
            ender = int(tmax / self.dts[np.mod(base + 1, 2)])
            if verbose:
                print('original length: ', len(self.motions[np.mod(base + 1, 2)]))
                print('trim length: ', ender)
            self.motions[np.mod(base + 1, 2)] = self.motions[np.mod(base + 1, 2)][:ender]
        return min_ind

    def generate_response_spectrums(self):
        for i in range(len(self.slaves) + 1):
            self.motions(i).generate_response_spectrum()

    def calculate_ratios(self):

        self.generate_response_spectrums()

        self.resp_log_ratio = np.log10(self.motions(1).s_a / self.motions(0).s_a)
        self.fa_log_ratio = np.log10(self.motions(1).smooth_fa_spectrum / self.motions(0).smooth_fa_spectrum)
        centre_freq = 1.0 / self.response_times[len(self.response_times) / 2]

        self.response_ratio_moment = sum(self.resp_log_ratio * (np.log10((1.0 /
                                         self.response_times) / centre_freq)))