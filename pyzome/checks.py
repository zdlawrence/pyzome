import re

import numpy as np
import xarray as xr

from typing import Union, List

from .exceptions import LongitudeError, CoordinateError, AttrError, UnitError


coord_regex = {
    "lon" : re.compile("lon[a-z]*"),
    "lat" : re.compile("lat[a-z]*"),
    "lev" : re.compile("(p?lev|pre?s|isobaric)[a-z]*")
}

var_units = {
    "wind" : ("m s-1", "m/s", "m / s" "meters per second"),
    "temperature" : ("K", "degK", "Kelvin"),
    "pressure" : ("Pa", "Pascals"),
    "vvel" : ("Pa s-1", "Pa/s", "Pa / s", "Pascals per second")
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
                    msg = f"Found multiple coordinates in dat matching the "+\
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

    if 'units' not in dat.attrs:
        msg = f'units is not an attribute of the given DataArray'
        raise AttrError(msg)

    units_SI = dat.units in var_units[var]
    if (enforce is True) and (units_SI is False):
        msg = f"The units '{dat.units}' do not match SI units for the {var}"+\
              f" category."
        raise UnitError(msg)

    return units_SI