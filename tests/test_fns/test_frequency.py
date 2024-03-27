import numpy as np

import eqsig
from tests.conftest import TEST_DATA_DIR


def test_fa_spectrum_conversion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    dt = 0.01
    values = np.loadtxt(record_path + record_filename, skiprows=2)

    npts = len(values)
    n_factor = 2 ** int(np.ceil(np.log2(npts)))
    fa = np.fft.fft(values, n=n_factor)
    points = int(n_factor / 2)
    fas = fa[range(points)] * dt
    faf = np.arange(points) / (2 * points * dt)
    n = 2 * len(fas)
    asig = eqsig.AccSignal(values, dt)
    fas_eqsig, faf_eqsig = eqsig.fns.frequency.generate_fa_spectrum(asig)

    assert np.isclose(fas, fas_eqsig).all()
    assert np.isclose(faf, faf_eqsig).all()

    a = np.zeros(len(fa), dtype=complex)
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    sig = np.fft.ifft(fa, n=n_factor)
    sig = sig[:len(values)]
    assert np.isclose(np.sum(np.abs(sig)), np.sum(np.abs(values)))
    asig2 = eqsig.fns.frequency.fas2signal(fas_eqsig, dt, stype="signal")
    trimmed = asig2.values[:len(values)]
    assert np.isclose(np.sum(np.abs(trimmed)), np.sum(np.abs(values)))
