.. image:: https://travis-ci.org/eng-tools/eqsig.svg?branch=master
   :target: https://travis-ci.org/eng-tools/eqsig
   :alt: Testing Status

.. image:: https://img.shields.io/pypi/v/eqsig.svg
   :target: https://pypi.python.org/pypi/eqsig
   :alt: PyPi version

.. image:: https://coveralls.io/repos/github/eng-tools/eqsig/badge.svg
   :target: https://coveralls.io/github/eng-tools/eqsig

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/eng-tools/eqsig/blob/master/LICENSE
    :alt: License

.. image:: https://eng-tools.github.io/static/img/ecp-badge.svg
    :target: https://eng-tools.github.io
    :alt: ECP project

*****
eqsig
*****

A Python package for seismic signal processing.

How to Use
==========

[Eqsig documentation](https://eqsig.readthedocs.io)

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

*

Contributing
============

How do I get set up?
--------------------

1. Run ``pip install -r requirements.txt``


Package conventions
-------------------

* A function that calculates a property the takes a signal as an input, should be named as `calc_<property>`,
 if the calculation has multiple different implementations, then include the citation
 as author and year as well `calc_<property>_<author>_<year>`
* If the function takes a raw array then it should contain the word array (or values).

Testing
-------

Tests are run with pytest

* Locally run: ``pytest`` on the command line.

* Tests are run on every push using travis, see the ``.travis.yml`` file


Deployment
----------

To deploy the package to pypi.com you need to:

 1. Push to the *pypi* branch. This executes the tests on circleci.com

 2. Create a git tag and push to github, run: ``trigger_deploy.py`` or manually:

 .. code:: bash

    git tag 0.5.2 -m "version 0.5.2"
    git push --tags origin pypi


Documentation
-------------

Built via Sphinx following: https://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/

For development mode

 1. cd to docs
 2. Run ``make html``

Docstrings follow numpy convention (in progress): https://numpydoc.readthedocs.io/en/latest/format.html

To fix long_description in setup.py: ``pip install collective.checkdocs``, ``python setup.py checkdocs``