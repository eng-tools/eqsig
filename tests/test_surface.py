import numpy as np
import eqsig

from tests import conftest


def _time_shifted_response_series(asig, times, up_red=1, down_red=1, add=False, trim=False):
    """
    Computes the time shifted response with reduction factors for the upward and downward motion

    :param asig:
    :param times:
    :param up_red:
    :param down_red:
    :return:
    """
    dt = asig.dt
    points = times / dt
    max_points = np.max(points)
    if trim:
        end = asig.npts
    else:
        end = None
    if hasattr(times, "__len__"):
        if not hasattr(up_red, "__len__"):
            up_red = up_red * np.ones_like(times)
        if not hasattr(down_red, "__len__"):
            down_red = down_red * np.ones_like(times)
        energy = []
        for tt, time in enumerate(times):
            ss = int(points[tt])
            extra = int(max_points - ss)
            up_acc = list(asig.values) + [0] * ss + [0] * extra
            down_acc = [0] * ss + list(asig.values) + [0] * extra
            if add:
                new_sig = np.array(up_acc) * up_red[tt] + np.array(down_acc) * down_red[tt]
            else:
                new_sig = np.array(up_acc) * up_red[tt] - np.array(down_acc) * down_red[tt]
            nsig = eqsig.AccSignal(new_sig, dt)
            ke = eqsig.im.calc_unit_kinetic_energy(nsig)
            energy.append(np.asarray(ke[:end]))
        return np.asarray(energy)
    else:
        ss = int(points)
        extra = int(max_points - ss)
        up_acc = list(asig.values) + [0] * ss + [0] * extra
        down_acc = [0] * ss + list(asig.values) + [0] * extra
        new_sig = np.array(up_acc) * up_red - np.array(down_acc) * down_red
        nsig = eqsig.AccSignal(new_sig, dt)
        ke = eqsig.im.calc_unit_kinetic_energy(nsig)[:end]
        return ke


def test_calc_cum_abs_surface_energy_w_sine_wave():

    b = np.sin(np.linspace(0, 10, 100))
    accsig = eqsig.AccSignal(b, 0.1)
    tshifts = np.array([0.1, 0.2])
    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, tshifts, nodal=True, up_red=1, down_red=1, stt=0.0,
                                                      trim=True, start=False)
    expected_cases = _time_shifted_response_series(accsig, 2 * tshifts, up_red=1, down_red=1, add=False, trim=True)
    assert len(cases[0]) == len(b)
    assert np.isclose(cases[0][-1], expected_cases[0][-1])
    assert np.isclose(cases[1][-1], expected_cases[1][-1])


def test_calc_cum_abs_surface_energy():
    accsig = conftest.t_asig()
    accsig = eqsig.AccSignal(accsig.values * 1e6, accsig.dt)
    tshifts = np.array([0, 0.01, 0.2, 1.5])
    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, tshifts, nodal=True, up_red=1, down_red=1, stt=0.,
                                                      trim=False, start=False)
    expected_cases = _time_shifted_response_series(accsig, 2 * tshifts, up_red=1, down_red=1, add=False, trim=True)
    assert np.isclose(cases[0][-1], 0.0)
    assert np.isclose(cases[0][-1], expected_cases[0][-1])
    assert np.isclose(cases[1][-1], expected_cases[1][-1])
    assert np.isclose(cases[2][-1], expected_cases[2][-1])
    assert np.isclose(cases[3][-1], expected_cases[3][-1])
    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, tshifts, nodal=True, up_red=1, down_red=1, stt=0.,
                                                      trim=True, start=False)
    assert np.isclose(cases[0][-1], 0.0)
    assert np.isclose(cases[0][-1], expected_cases[0][-1])
    assert np.isclose(cases[1][-1], expected_cases[1][-1])
    assert np.isclose(cases[2][-1], expected_cases[2][-1])
    assert np.isclose(cases[3][-1], expected_cases[3][-1])


def test_calc_cum_abs_surface_energy_start_check_same_energy():
    accsig = conftest.t_asig()
    tshifts = np.array([0.01, 0.2, 0.5])
    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, tshifts, nodal=True, up_red=1, down_red=1, stt=0.0,
                                                      trim=False, start=False)
    cases_w_s = eqsig.surface.calc_cum_abs_surface_energy(accsig, tshifts, nodal=True, up_red=1, down_red=1, stt=0.0,
                                                      trim=False, start=True)
    diff = np.sum(abs(cases[:, -1] - cases_w_s[:, -1]))
    assert diff < 2.0e-5, diff


def test_calc_cum_abs_surface_energy_start_from_base():
    accsig = conftest.t_asig()
    travel_times = np.array([0.01, 0.2, 0.5])
    stt = 1.0
    cases_w_s = eqsig.surface.calc_cum_abs_surface_energy(accsig, travel_times, nodal=True, up_red=1, down_red=1,
                                                          stt=stt, trim=True, start=True)

    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, travel_times, nodal=True, up_red=1, down_red=1, stt=stt,
                                                      trim=True, start=False)
    case_interp_0 = np.interp(accsig.time + (stt - travel_times[0]), accsig.time, cases_w_s[0])
    diff0 = np.sum(abs(case_interp_0 - cases[0])) / cases[0][-1]
    assert np.isclose(diff0, 0., atol=1.0e-7), diff0
    case_interp_1 = np.interp(accsig.time + (stt - travel_times[1]), accsig.time, cases_w_s[1])
    diff1 = np.sum(abs(case_interp_1 - cases[1])) / cases[1][-1]
    assert np.isclose(diff1, 0., atol=1.0e-7), diff1
    case_interp_2 = np.interp(accsig.time + (stt - travel_times[2]), accsig.time, cases_w_s[2])
    diff2 = np.sum(abs(case_interp_2 - cases[2])) / cases[2][-1]
    assert np.isclose(diff2, 0., atol=1.0e-7), diff2


def test_calc_cum_abs_surface_energy_start_from_top():
    accsig = conftest.t_asig()
    travel_times = np.array([0.01, 0.2, 0.5])
    stt = 0.0
    cases_w_s = eqsig.surface.calc_cum_abs_surface_energy(accsig, travel_times, nodal=True, up_red=1, down_red=1,
                                                          stt=stt, trim=True, start=True)

    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, travel_times, nodal=True, up_red=1, down_red=1, stt=stt,
                                                      trim=True, start=False)
    case_interp_0 = np.interp(accsig.time + (stt - travel_times[0]), accsig.time, cases_w_s[0])
    diff0 = np.sum(abs(case_interp_0 - cases[0])) / cases[0][-1]
    assert np.isclose(diff0, 0., atol=5.0e-3), diff0
    case_interp_1 = np.interp(accsig.time + (stt - travel_times[1]), accsig.time, cases_w_s[1])
    diff1 = np.sum(abs(case_interp_1 - cases[1])) / cases[1][-1]
    assert np.isclose(diff1, 0., atol=5.0e-3), diff1
    case_interp_2 = np.interp(accsig.time + (stt - travel_times[2]), accsig.time, cases_w_s[2])
    diff2 = np.sum(abs(case_interp_2 - cases[2])) / cases[2][-1]
    assert np.isclose(diff2, 0., atol=8.0e-2), diff2



def skip_plot():
    import matplotlib.pyplot as plt
    accsig = eqsig.load_asig(conftest.TEST_DATA_DIR + 'test_motion_dt0p01.txt')
    accsig = eqsig.AccSignal(accsig.values[100:], accsig.dt)
    travel_times = np.linspace(0, 2, 44)
    travel_times = np.array([0.2, 0.5])
    stt = 0.0
    cases_w_s = eqsig.surface.calc_cum_abs_surface_energy(accsig, travel_times, nodal=True, up_red=1, down_red=1, stt=stt,
                                                      trim=True, start=True)
    # plt.plot(tshifts, cases[:, -1])
    # plt.plot(tshifts, expected_cases[:, -1])
    from bwplot import cbox
    plt.plot(accsig.time, cases_w_s[0], c=cbox(0))
    plt.plot(accsig.time, cases_w_s[1], c='r')
    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, travel_times, nodal=True, up_red=1, down_red=1, stt=stt,
                                                      trim=True, start=False)
    plt.plot(accsig.time, cases[0], c=cbox(1), ls='--')
    plt.plot(accsig.time, cases[1], c=cbox(1), ls='--')
    plt.plot(accsig.time + (stt - travel_times[0]), cases[0], c='k', ls=':')
    plt.plot(accsig.time + (stt - travel_times[1]), cases[1], c='k', ls=':')
    case_interp_0 = np.interp(accsig.time + (stt - travel_times[0]), accsig.time, cases_w_s[0])
    diff0 = np.sum(abs(case_interp_0 - cases[0])) / cases[0][-1]
    assert np.isclose(diff0, 0., atol=5.0e-3), diff0
    case_interp_1 = np.interp(accsig.time + (stt - travel_times[1]), accsig.time, cases_w_s[1])
    diff1 = np.sum(abs(case_interp_1 - cases[1])) / cases[1][-1]
    assert np.isclose(diff1, 0., atol=5.0e-3), diff1
    case_interp_2 = np.interp(accsig.time, accsig.time + (stt - travel_times[2]), cases_w_s[2])
    diff2 = np.sum(abs(case_interp_2 - cases[2])) / cases[2][-1]
    assert np.isclose(diff2, 0., atol=5.0e-3), diff2

    plt.show()

def skip_plot2():
    import matplotlib.pyplot as plt
    asig = eqsig.load_asig(conftest.TEST_DATA_DIR + 'test_motion_dt0p01.txt')
    travel_times = np.array([0.2, 0.5])
    cases = eqsig.surface.calc_cum_abs_surface_energy(asig, travel_times, nodal=True, up_red=1, down_red=1, stt=1.0,
                                                      trim=False, start=True)
    # plt.plot(tshifts, cases[:, -1])
    # plt.plot(tshifts, expected_cases[:, -1])
    plt.plot(cases[0])
    plt.plot(cases[1])

    cases = eqsig.surface.calc_cum_abs_surface_energy(asig, travel_times, nodal=True, up_red=1, down_red=1, stt=1.0,
                                                      trim=False, start=False)
    plt.plot(cases[0])
    plt.plot(cases[1])

    # plt.plot(asig.time - 2 * travel_times[0], cases[0])
    # plt.plot(asig.time - 2 * travel_times[1], cases[1])
    # plt.plot(asig.time - 2 * travel_times[2], cases[2])
    plt.show()


def test_calc_surface_energy_array_sizing():
    asig = eqsig.AccSignal(np.arange(10), dt=0.5)
    travel_times = np.array([0.0, 0.5])
    vals0 = eqsig.surface.calc_surface_energy(asig, travel_times, nodal=True, up_red=1., down_red=1., stt=0.0, trim=True)
    vshape = vals0.shape
    assert vshape[0] == 2
    assert vshape[1] == 10
    cvals0 = eqsig.surface.calc_cum_abs_surface_energy(asig, travel_times, nodal=True, up_red=1., down_red=1., stt=0.0, trim=True)
    vshape = cvals0.shape
    assert vshape[0] == 2
    assert vshape[1] == 10

    travel_times = np.array([0.0, 0.5])
    up_red = np.array([1., 0.9])
    down_red = np.array([1., 0.9])
    vals = eqsig.surface.calc_surface_energy(asig, travel_times, nodal=True, up_red=up_red, down_red=down_red,
                                             stt=0.0, trim=True)
    vshape = vals.shape
    assert vshape[0] == 2
    assert vshape[1] == 10
    vals = eqsig.surface.calc_cum_abs_surface_energy(asig, travel_times, nodal=True, up_red=up_red, down_red=down_red,
                                             stt=0.0, trim=True)
    vshape = vals.shape
    assert vshape[0] == 2
    assert vshape[1] == 10

    travel_times = 0.5
    vals2 = eqsig.surface.calc_surface_energy(asig, travel_times, nodal=True, up_red=1., down_red=1., stt=0.0, trim=True)
    vshape = vals2.shape
    assert vshape[0] == 10
    assert len(vshape) == 1
    cvals2 = eqsig.surface.calc_cum_abs_surface_energy(asig, travel_times, nodal=True, up_red=1., down_red=1., stt=0.0, trim=True)
    vshape = cvals2.shape
    assert vshape[0] == 10
    assert len(vshape) == 1
    assert np.isclose(np.sum(np.abs(vals0[1] - vals2)), 0.0)
    assert np.isclose(np.sum(np.abs(cvals0[1] - cvals2)), 0.0)


if __name__ == '__main__':
    # test_calc_cum_abs_surface_energy_start_check_same_energy()
    test_calc_cum_abs_surface_energy()
    # skip_plot()
    # test_calc_cum_abs_surface_energy()

