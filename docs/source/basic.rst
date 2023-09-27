.. _basic:

Basic operations (``pyzome.basic``)
=====================================
These functions extend operations that can be done with xarray, 
but provide your code with some additional readability, 
convenience, and safety. For instance, while::

   >>> ds.mean("lon")

will take the mean over the "lon" dimension regardless of 
what ``lon`` contains::

   >>> import pyzome as pzm
   >>> pzm.zonal_mean(ds)

will infer the dimension name, and check to ensure that the 
corresponding longitude dimension is regularly spaced and spans 
a full 360 degrees. The check on the longitudes can be disabled
by passing ``strict=False`` as a keyword argument.

The functions in this module are also provided as pyzome xarray 
accessor methods. The above example could also be achieved with:

   >>> ds.pzm.zonal_mean()

Functions
---------
.. automodule:: pyzome.basic
   :members: