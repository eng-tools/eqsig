*****
eqsig
*****

A Python package for seismic signal processing.

How to Use
==========


Examples
--------

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


Useful material
===============
