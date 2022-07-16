import re

import numpy as np
import xarray as xr

from typing import Union, List

from .exceptions import LongitudeError, CoordinateError, AttrError, UnitError


coord_regex = {
    "lon": re.compile("lon[a-z]*"),
    "lat": re.compile("lat[a-z]*"),
    "lev": re.compile("(p?lev|pre?s|isobaric)[a-z]*")
}

var_units = {
    "wind": ("m s-1", "m/s", "m / s" "meters per second"),
    "temperature": ("K", "degK", "Kelvin"),
    "vvel": ("Pa s-1", "Pa/s", "Pa / s", "Pascals per second"),
    "pressure": ("Pa", "Pascals"),
    "altitude": ("m", "meters")
}


def has_global_regular_lons(lons: Union[xr.DataArray, np.array],
                            enforce: bool = False) -> bool:

    # generally for regularly gridded data, the difference
    # between the max lon and min lon plus the lon delta
    # should equal 360 degrees
    span = float(lons.max() - lons.min())
    dlon = float(lons[1] - lons[0])

    # check that all the longitude differences are close to
    # the single dlon, to within 1.0e-3 degrees
    equal_spacing = np.allclose(np.diff(lons), dlon, rtol=0.0, atol=1.0e-3)

    # now check that the span+londelta cover 360 degrees
    full_span = (span + dlon) >= 360.

    if (enforce is True) and (equal_spacing is False):
        msg = 'longitudes are not equally spaced'
        raise LongitudeError(msg)
    elif (enforce is True) and (full_span is False):
        msg = 'longitudes do not span all 360 degrees'
        raise LongitudeError(msg)

    return (equal_spacing and full_span)


def infer_xr_coord_names(dat: Union[xr.DataArray, xr.Dataset],
                         required: List[str] = []) -> dict:

    coord_names = {}
    dat_coords = list(dat.coords)
    for dc in dat_coords:
        for coord in coord_regex:
            if coord_regex[coord].match(dc.lower()):
                # Check if more than one coord in dat matches the same pattern
                if coord in coord_names:
                    msg = "Found multiple coordinates in dat matching the " + \
                          f"pattern for '{coord}: {coord_names[coord]} & {dc}"
                    raise CoordinateError(msg)
                else:
                    coord_names[coord] = dc

    for req in required:
        if req not in coord_names:
            msg = f"Unable to match any of the coordinates in dat for {req}"
            raise CoordinateError(msg)

    return coord_names


def check_var_SI_units(dat: xr.DataArray, var: str,
                       enforce: bool = False) -> bool:

    if "units" not in dat.attrs:
        msg = "units is not an attribute of the given DataArray"
        raise AttrError(msg)

    units_SI = dat.units in var_units[var]
    if (enforce is True) and (units_SI is False):
        msg = f"The units '{dat.units}' do not match SI units for the {var}" + \
              " category."
        raise UnitError(msg)

    return units_SI


def check_for_logp_coord(dat: xr.DataArray, enforce: bool = False) -> bool:
    ### TO DO: Add tests

    if ("z" not in dat.coords):
        if (enforce is True):
            msg = "z is not a coordinate in the data"
            raise CoordinateError(msg)
        return False

    if (not {"units", "long_name"} <= dat.z.attrs.keys()):
        if (enforce is True):
            msg = "z is missing either units and/or long_name attributes"
            raise AttrError(msg)
        return False

    if (dat.z.attrs["long_name"] != "log-pressure altitude"):
        if (enforce is True):
            msg = "z must have a long_name = 'log-pressure altitude'"
            raise AttrError(msg)
        return False

    return check_var_SI_units(dat.z, "altitude", enforce=enforce)
