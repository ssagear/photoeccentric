.. _quicktest:

Getting started
===================

Two ways of using juliet
-------------------------

In the spirit of accomodating the code for everyone to use, ``juliet`` can be used in two different ways: as
an **imported library** and also in **command line mode**. Both give rise to the same results because the command
line mode simply calls the ``juliet`` libraries in a python script.

To use ``juliet`` as an **imported library**, inside any python script you can simply do:

.. code-block:: python

    import juliet
    dataset = juliet.load(priors = priors, t_lc=times, y_lc=flux, yerr_lc=flux_error)
    results = dataset.fit()

In this example, ``juliet`` will perform a fit on a lightcurve dataset defined by a dictionary of times ``times``,
relative fluxes ``flux`` and error on those fluxes ``flux_error`` given some prior information ``priors`` which,
as we will see below, is also defined through a dictionary.


In **command line mode**, ``juliet`` can be used through a simple call in any terminal. To do this, after
installing juliet, you can from anywhere in your system simply do:

.. code-block:: bash

    juliet -flag1 -flag2 --flag3

In this example, ``juliet`` is performing a fit using different inputs defined by ``-flag1``, ``-flag2`` and ``--flag3``.
There are several flags that can be used to accomodate your ``juliet`` runs through command-line which we'll explore
in the tutorials. There is a third way of using ``juliet``, which is by calling the ``juliet.py`` code and applying
these same flags (as it is currently explained in `project's wiki page <https://github.com/nespinoza/juliet/wiki>`_).
However, no further updates will be done for that method, and the ones defined above should be the preferred ones to
use.

A first fit to data with juliet
-----------------------------------------------

To showcase how ``juliet`` works, let us first perform an extremely simple fit to data using ``juliet`` as an *imported library*.
We will fit the TESS data of TOI-141 b, which was shown to host a 1-day transiting exoplanet by
`Espinoza et al. (2019) <https://arxiv.org/abs/1903.07694>`_. Let us first load the data corresponding to this
object, which is hosted in MAST. For TESS data, ``juliet`` has already built-in functions to load the data arrays
directly given a web link to the data --- let's load it and plot the data to see how it looks:
