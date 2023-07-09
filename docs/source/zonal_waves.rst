.. _zonal_waves:

Zonal Wavenumber Decompositions (``pyzome.zonal_waves``)
========================================================
These functions provide utilities for decomposing fields 
as a function of zonal wavenumber. The primary function 
of interest is :py:func:`pyzome.zonal_wave_coeffs`, which 
performs a rFFT operation across the longitude dimension 
of the input, returning the complex Fourier coefficients. 
These coefficients can then be used in the other functions 
described here to, e.g., filter the original field, find 
the amplitude/phases of the waves, etc.

:py:func:`pyzome.zonal_wave_coeffs` and some of the other 
functions herein can use one of two modules to perform the 
forward/inverse Fourier transforms, either ``"scipy"`` or 
``"xrft"``. These are provided as separate options because:

   * ``scipy`` FFTs output a dtype consistent with its input (e.g., ``complex64`` for ``float32`` input, and ``complex128`` for ``float64`` input) BUT works in a memory-eager manner (i.e., it loads the input xarray data)
   * ``xrft`` FFTs always output ``complex128`` BUT they are able to operate lazily and in parallel using ``xarray`` and ``dask``.

``scipy`` is used by default. In the future, these options 
may be deprecated when/if a better solution is developed 
that combines the benefits of both without explicit 
loading or type-conversion. 

Functions
---------
.. automodule:: pyzome.zonal_waves
   :members: