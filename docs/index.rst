.. eqsig documentation master file, created by
   sphinx-quickstart on Tue Jul 10 19:21:51 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to eqsig's documentation!
=================================

Release v\ |version|

The eqsig package is for signal processing of earthquake related data.

It provides two main objects the ``Signal`` and ``AccSignal`` objects that represent time series data of any type
and acceleration time series data.

The objects have common signal processing methods available (e.g. Butterworth filtering) and common
parameters (eg. Fourier Spectra).

Most functions and intensity measures have been written using the signal objects as inputs but have also be seperately
defined to allow inputs as a numpy array and time step, these functions typically have ``_array`` in the function name.

Features
--------

* Compute the acceleration response spectrum and elastic response time series using the fast Nigam and Jennings (1968) algorithm.
* Compute the Fourier amplitude spectrum (using the scipy.signal.fft algorithm)
* Compute the smooth Fourier amplitude spectrum according to Konno and Ohmachi (1998)
* Compute velocity and displacement from acceleration time series
* Compute peak ground motion quantities (PGA, PGV, PGD)
* Compute common ground motion intensity measures (Arias intensity, CAV, CAV_dp5, significant duration, bracketed duration, dominant period)
* Compute signal features (zero crossings, global peaks, local peaks)
* Compute rotated ground motion or intensity measure from two ground motion components
* Resampling of ground motion through interpolation or periodic resampling
* Butterworth filter (using scipy), running average, polynomial fitting
* Fast loading of, and saving of, plain text to and from Signal objects

Package contents
----------------

.. toctree::
   :maxdepth: 2

   eqsig
   install
   examples

* :ref:`genindex`

Example
-------


Generate response spectra

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import eqsig.single

    bf, sub_fig = plt.subplots()
    a = np.loadtxt("<path-to-acceleration-time-series>")
    dt = 0.005  # time step of acceleration time series
    periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
    record = eqsig.AccSignal(a * 9.8, dt)
    record.generate_response_spectrum(response_times=periods)
    times = record.response_times

    sub_fig.plot(times, record.s_a, label="eqsig")
    plt.show()

