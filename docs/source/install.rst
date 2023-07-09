.. _install:

Installation instructions
=========================

Dependencies
------------

``pyzome`` currently only requires `numpy <http://www.numpy.org/>`__, 
`scipy <https://www.scipy.org/>`__, `xarray <http://xarray.pydata.org/en/stable/>`__, 
and `xrft <https://xrft.readthedocs.io/en/latest/>`__.

Installation
------------

``pyzome`` is not yet available on PyPI, but it can be
installed by cloning its Github repository (or downloading the 
source code) and running::

    python setup.py install

in the top-level pyzome directory. Alternatively, you can use pip 
to install from the cloned/source directory with::

    pip install -e .

This will install ``pyzome`` in your current Python environment.