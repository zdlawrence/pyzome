import numpy as np
import xarray as xr


def _make_regular_coord(
    low: float | int,
    high: float | int,
    step: float | int,
    name: str,
    inclusive: bool = False,
) -> xr.DataArray:
    """
    Generate an xr.DataArray representing a regularly spaced coordinate array 

    Parameters
    ----------
    low : float | int   
        The lower limit of the coordinate
    high : float | int
        The upper limit of the coordinate
    step : float | int
        The step size between each element of the coordinate
    name : str
        The name of the coordinate  
    inclusive : bool, optional
        Whether to include the upper limit in the coordinate
    
    Returns
    -------
    `xarray.DataArray`

    """
    coord = np.arange(low, high + (inclusive * step), step).astype("float64")
    return xr.DataArray(coord, dims=[name], name=name).assign_coords({name: coord})


def lat_coord(
    resolution: float | int,
    name: str = "lat",
    low_lim: float | int = -90.0,
    upp_lim: float | int = 90.0,
    inclusive: bool = True,
) -> xr.DataArray:
    """
    Generate a regularly spaced latitude-like coordinate

    Parameters
    ----------
    resolution : float | int
        The spacing between latitudes in degrees
    name : str, optional
        The name of the coordinate, defaults to "lat"
    low_lim : float | int, optional
        The lower latitude limit, which defaults to -90
    upp_lim : float | int, optional
        The upper latitude limit, which defaults to 90
    inclusive : bool, optional
        Whether to include the upper latitude limit, which  defaults to True
    
    Returns
    -------
    `xarray.DataArray` of latitudes

    """
    if (low_lim >= upp_lim) or (low_lim < -90) or (upp_lim > 90):
        msg = f"Invalid latitudes request: low_lim={low_lim}, upp_lim={upp_lim}"
        raise ValueError(msg)

    lats = _make_regular_coord(low_lim, upp_lim, resolution, name, inclusive=inclusive)
    lats.attrs["standard_name"] = "latitude"
    lats.attrs["units"] = "degrees_north"

    return lats


def lon_coord(
    resolution: float | int,
    name: str = "lon",
    low_lim: float | int = 0.0,
    upp_lim: float | int = 360.0,
    inclusive: bool = False,
) -> xr.DataArray:
    """
    Generate a regularly spaced longitude-like coordinate

    Parameters
    ----------
    resolution : float | int
        The spacing between longitudes in degrees
    name : str, optional
        The name of the coordinate, defaults to "lon"
    low_lim : float | int, optional
        The lower longitude limit, which defaults to 0
    upp_lim : float | int, optional
        The upper longitude limit, which defaults to 360
    inclusive : bool, optional
        Whether to include the upper longitude limit, which  defaults to False
    
    Returns
    -------
    `xarray.DataArray` of longitudes

    """
    if low_lim >= upp_lim:
        msg = "low_lim >= upp_lim; invalid limits!"
        raise ValueError(msg)

    if (low_lim < 0) and (upp_lim > 180):
        msg = "Proper longitude range should be within [-180,180] or [0, 360]"
        raise ValueError(msg)
    elif upp_lim > 360:
        msg = "upper longitude limit cannot exceed 360"
        raise ValueError(msg)
    elif low_lim < -180:
        msg = "lower longitude limit cannot be below -180"
        raise ValueError(msg)

    lons = _make_regular_coord(low_lim, upp_lim, resolution, name, inclusive=inclusive)
    lons.attrs["standard_name"] = "longitude"
    lons.attrs["units"] = "degrees_east"

    return lons


def plev_coord(
    levels_per_decade: int,
    low_lim_exponent: int = 3,
    upp_lim_exponent: int = 0,
    name: str = "lev",
    units: str = "hPa",
) -> xr.DataArray:
    """
    Generate a pressure-like vertical coordinate with logarithmic spacing

    Parameters
    ----------
    levels_per_decade : int
        The number of levels per decade (i.e., between 10^3 and 10^2, 10^2 and 10^1, etc.)
    low_lim_exponent : int, optional
        The base-10 exponent for the lowest pressure level, which defaults to 3 (for 1000 hPa)
    upp_lim_exponent : int, optional
        The base-10 exponent for the highest pressure level, which defaults to 0 (for 1 hPa)
    name : str, optional
        The name of the coordinate, which defaults to "lev"
    units : str, optional
        The units of the coordinate, which defaults to "hPa"
    
    Returns
    -------
    `xarray.DataArray` of pressure levels

    """
    num_elements = levels_per_decade * np.abs(low_lim_exponent - upp_lim_exponent) + 1
    pres = np.logspace(low_lim_exponent, upp_lim_exponent, num_elements)
    pres = xr.DataArray(pres, dims=[name], name=name).assign_coords({name: pres})
    pres.attrs["standard_name"] = "pressure"
    pres.attrs["units"] = units

    return pres


def time_coord(
    start: str = "2000-01-01",
    end: str = "2001-01-01",
    freq: str = "M",
) -> xr.DataArray:
    """
    Generate a time-like coordinate

    Parameters
    ----------
    start : str, optional
        The start date, which defaults to "2000-01-01"
    end : str, optional
        The end date, which defaults to "2001-01-01"
    freq : str, optional
        The frequency of the time coordinate, which defaults to "M" (monthly). 
        See https://numpy.org/doc/stable/reference/arrays.datetime.html#basic-datetimes
        for more information.
    
    Returns
    -------
    `xarray.DataArray` of times

    """
    times = np.arange(start, end, dtype=f"datetime64[{freq}]").astype("datetime64[ns]")
    times = xr.DataArray(times, dims=["time"], name="time").assign_coords({"time": times})
    times.attrs["standard_name"] = "time"
    times.attrs["units"] = f"days since {start}"

    return times


def create_dummy_geo_field(
    lons: xr.DataArray, 
    lats: xr.DataArray, 
    levs: None | xr.DataArray = None, 
    times: None | xr.DataArray = None,
    name: str = "dummy",
    attrs: dict = {},
) -> xr.DataArray:
    """
    Mock a geophysical field with random data

    Parameters
    ----------
    lons : `xarray.DataArray`
        The longitudes of the data
    lats : `xarray.DataArray`
        The latitudes of the data
    levs : `xarray.DataArray`, optional
        The vertical levels of the data. Defaults to None (vertical levels excluded)
    times : `xarray.DataArray`, optional
        The times of the data. Defaults to None (times excluded)
    name : str, optional
        The name of the data. Defaults to "dummy"
    attrs : dict, optional
        The attributes of the data. Defaults to an empty dictionary.
    
    Returns
    -------
    `xarray.DataArray` of random data with corresponding geophysical coordinates

    """

    dat_shape = [lats.size, lons.size]
    coords = [lats, lons]
    dims = ["lat", "lon"]
    
    if (levs is not None):
        dat_shape.insert(0, levs.size)
        coords.insert(0, levs)
        dims.insert(0, "lev")
    
    if (times is not None):
        dat_shape.insert(0, times.size)
        coords.insert(0, times)
        dims.insert(0, "time")
    
    dummy = np.random.normal(size=dat_shape).astype("float32")
    da = xr.DataArray(dummy, coords=coords, dims=dims, name=name)
    da.attrs = attrs
    return da


def create_dummy_geo_dataset(
    field_names: list[str],
    lons: xr.DataArray,
    lats: xr.DataArray,
    levs: None | xr.DataArray = None,
    times: None | xr.DataArray = None,
    field_attrs: None | dict = None,
) -> xr.Dataset:
    """
    Mock a dataset of geophysical fields with random data

    Parameters
    ----------
    field_names : list[str]
        The names of the fields
    lons : `xarray.DataArray`
        The longitudes of the data
    lats : `xarray.DataArray`
        The latitudes of the data
    levs : `xarray.DataArray`, optional
        The vertical levels of the data. Defaults to None (vertical levels excluded)
    times : `xarray.DataArray`, optional
        The times of the data. Defaults to None (times excluded)
    field_attrs : dict, optional
        The attributes of the data. Defaults to None. If provided, this should be 
        a nested dictionary with the outer keys corresponding to the field names,
        and inner keys/dicts corresponding to the attributes for each field.

    """
    if field_attrs is not None and len(field_attrs) != len(field_names):
        raise ValueError(
            f"Number of field names ({len(field_names)}) must match number of field attributes ({len(field_attrs)})"
        )
    ds = xr.Dataset()
    for name in field_names:
        da = create_dummy_geo_field(lons, lats, levs, times, name)
        ds[name] = da
        if field_attrs is not None:
            ds[name].attrs = field_attrs[name]
    return ds
