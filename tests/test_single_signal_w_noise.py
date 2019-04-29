import numpy as np
from eqsig.measures import calc_sig_dur
from tests.conftest import TEST_DATA_DIR, load_test_record_from_file


def test_remove_polyfit_1(noise_rec):
    noise_rec.remove_poly(poly_fit=1)
    ssq_org = np.sum(noise_rec.values ** 2)

    noise_rec.add_series(np.linspace(0, 0.2, noise_rec.npts))
    ssq_w_linear = np.sum(noise_rec.values ** 2)

    noise_rec.remove_poly(poly_fit=1)
    ssq_corrected = np.sum(noise_rec.values ** 2)

    assert ssq_org != ssq_w_linear
    assert np.isclose(ssq_org, ssq_corrected)


def test_remove_polyfit_2(noise_rec):
    noise_rec.remove_poly(poly_fit=2)
    ssq_org = np.sum(noise_rec.values ** 2)

    x = np.linspace(0, 1.0, noise_rec.npts)
    noise_rec.add_series(0.2 * x - 0.5 * x ** 2)
    ssq_w_linear = np.sum(noise_rec.values ** 2)

    noise_rec.remove_poly(poly_fit=2)
    ssq_corrected = np.sum(noise_rec.values ** 2)

    assert ssq_org != ssq_w_linear
    assert np.isclose(ssq_org, ssq_corrected)


def test_butterpass(noise_rec):
    ssq_org = np.sum(noise_rec.values ** 2)

    x = np.linspace(0, 1.0, noise_rec.npts)
    noise_rec.add_series(0.2 * x - 0.5 * x ** 2)
    ssq_w_linear = np.sum(noise_rec.values ** 2)

    noise_rec.butter_pass([0.2, 25])
    ssq_corrected = np.sum(noise_rec.values ** 2)

    assert ssq_org != ssq_w_linear
    assert np.isclose(ssq_org, 31854.72888, rtol=0.0001)
    assert np.isclose(ssq_corrected, 10663.4862479, rtol=0.0001)


def test_fourier_spectra(noise_rec):
    assert len(noise_rec.fa_spectrum) > 1
    assert len(noise_rec.fa_frequencies) > 1


def test_duration_stats(noise_rec):
    t_595 = calc_sig_dur(noise_rec)
    assert np.isclose(t_595, 18.025)


