import numpy as np
import xarray as xr


def create_dummy_xr_data(dims):

    coords = []
    shape = []
    for dim in dims:
        if (dim in ["lon","longitude"]):
            lon = np.arange(0,360,5)
            coords.append(lon)
            shape.append(lon.size)
        elif (dim in ["lat","latitude"]):
            lat = np.arange(-90,90.1,5)
            coords.append(lat)
            shape.append(lat.size)
        elif (dim in ["lev","level"]):
            lev = np.array([1000,500,300,100,50,30,10])
            coords.append(lev)
            shape.append(lev.size)
        else:
            coords.append(np.arange(5))
            shape.append(5)

    dummy = np.random.normal(size=shape).astype("float32")
    da = xr.DataArray(dummy, coords=coords, dims=dims)

    return da