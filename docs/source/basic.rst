.. _basic:

Basic operations (``pyzome.basic``)
=====================================
These functions extend operations that can be done with xarray, 
but provide your code with some additional readability, 
convenience, and safety. For instance, while::

   >>> ds.mean("lon")

will take the mean over the "lon" dimension regardless of 
what ``lon`` contains::

   >>> pzm.zonal_mean(ds, strict=True)

will infer the dimension name, and check to ensure that the 
corresponding longitude dimension is regularly spaced and spans 
a full 360 degrees. 

Functions
---------
.. automodule:: pyzome.basic
   :members: