.. _api:

API
====
.. module:: photoeccentric

The core classes within ``photoeccentric`` are the ``KeplerStar`` and ``KOI`` classes. After creating  a ``photoeccentric.KeplerStar`` object with information about the host star of a KOI system,
you will be able to create a ``photoeccentric.KOI`` object for each planet you wish to investigate. This ``photoeccentric.KOI`` object will allow you to fit and save information about the
fit Kepler light curve, eccentricity posteriors, etc...)

The KOI object defines the planet in question.

.. autoclass:: photoeccentric.KeplerStar
   :members:

.. autoclass:: photoeccentric.KOI
   :members:

.. automodule:: photoeccentric.eccentricity
   :members:

.. automodule:: photoeccentric.lcfitter
   :members:

.. automodule:: photoeccentric.stellardensity
   :members:

.. automodule:: photoeccentric.utils
   :members:
