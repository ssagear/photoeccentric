Usage
=====

.. _installation:

Installation
------------

To use photoeccentric, first install it using dist-utils:

.. code-block:: console

   (.venv) $ python setup.py install

Using photoeccentric
----------------
..
To retrieve a list of random ingredients,
..
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

..
The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
..
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
..
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import photoeccentric as ph
..
>>> lumache.get_random_ingredients()
..
['shells', 'gorgonzola', 'parsley']
