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

:py:func:`pyzome.zonal_wave_coeffs` requires that the 
underlying input have longitudes that are regularly 
spaced and span the globe. This way the FFTs can be 
done with no windowing/tapering, since the domain is 
periodic. 

:py:func:`pyzome.zonal_wave_coeffs` and some of the other 
functions herein can use one of two modules to perform the 
forward/inverse Fourier transforms, either ``scipy`` or 
``xrft``. ``scipy`` is used by default since its 
FFT functions output dtypes consistent with the input 
(e.g., ``complex64`` for ``float32`` input, and ``complex128``
for ``float64`` input). Both backends are able to operate 
lazily by leveraging ``dask``. These options may be deprecated
in the future in favor of using a single backend alone. 


Functions
---------
.. automodule:: pyzome.zonal_waves
   :members: