import numpy as np

from eqsig import im, functions

from tests import conftest


def test_cavdp():
    asig = conftest.t_asig()

    cav_dp = im.calc_cav_dp(asig)[-1]

    # 0.810412819 from eqsig 0.5.26  tested with several motions
    assert np.isclose(cav_dp, 0.81041282, rtol=0.001), cav_dp


def test_calc_cyclic_amp_combined_arrays_w_power_law():
    asig = conftest.t_asig()
    csr_array0 = asig.values
    csr_array1 = asig.values
    b = 0.34
    csr_n15_series = im.calc_cyc_amp_combined_arrays_w_power_law(csr_array0, csr_array1, n_cyc=15, b=b)
    a1_peak_inds_end = functions.get_switched_peak_array_indices(csr_array0)
    a1_csr_peaks_end = np.abs(np.take(csr_array0, a1_peak_inds_end))

    a2_peak_inds_end = functions.get_switched_peak_array_indices(csr_array1)
    a2_csr_peaks_end = np.abs(np.take(csr_array1, a2_peak_inds_end))

    all_csr_peaks_end = np.array(list(a1_csr_peaks_end) + list(a2_csr_peaks_end))
    csr_n15_end = np.sum((np.abs(all_csr_peaks_end) ** (1. / b)) / 2 / 15) ** b
    assert np.isclose(csr_n15_series[-1], csr_n15_end), (csr_n15_series[-1], csr_n15_end)
    assert np.isclose(csr_n15_series[-1], 1.2188083, rtol=1.0e-4)  # v=1.1.2


def test_n_cyc_and_cyc_amplitude():
    asig = conftest.t_asig()
    csr_array0 = asig.values

    n_cyc = im.calc_n_cyc_array_w_power_law(csr_array0, a_ref=asig.pga * 0.65, b=0.3, cut_off=0.01)
    csr = im.calc_cyc_amp_array_w_power_law(csr_array0, n_cyc=n_cyc[-1], b=0.3)
    assert np.isclose(csr[-1], asig.pga * 0.65), (csr[-1], asig.pga * 0.65)
    assert np.isclose(n_cyc[-1], 17.23, rtol=1.0e-3), n_cyc[-1]   # v=1.1.2
    assert np.isclose(csr[-1], 0.9165), csr[-1]  # v=1.1.2


def test_calc_cyclic_amp_gm_arrays_w_power_law():
    asig = conftest.t_asig()
    csr_array0 = asig.values
    csr_array1 = asig.values
    b = 0.34
    csr_n15_series = im.calc_cyc_amp_gm_arrays_w_power_law(csr_array0, csr_array1, n_cyc=15, b=b)
    assert np.isclose(csr_n15_series[-1], 0.96290891), (csr_n15_series[-1], 0.96290891)  # v=1.1.2


if __name__ == '__main__':
    test_n_cyc_and_cyc_amplitude()