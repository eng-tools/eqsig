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
    tshifts = np.array([0.01, 0.2, 0.5])
    cases = eqsig.surface.calc_cum_abs_surface_energy(accsig, tshifts, nodal=True, up_red=1, down_red=1, stt=0.0,
                                                      trim=True, start=False)
    expected_cases = _time_shifted_response_series(accsig, 2 * tshifts, up_red=1, down_red=1, add=False, trim=True)
    assert len(cases[0]) == accsig.npts
    assert np.isclose(cases[0][-1], expected_cases[0][-1])
    assert np.isclose(cases[1][-1], expected_cases[1][-1])
    assert np.isclose(cases[2][-1], expected_cases[2][-1])


if __name__ == '__main__':
    test_calc_cum_abs_surface_energy()

