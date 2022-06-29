import re
from typing import Union, List
from xarray import DataArray, Dataset

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


def infer_xr_coord_names(dat: Union[DataArray, Dataset],
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
                    raise ValueError(msg)
                else:
                    coord_names[coord] = dc

    for req in required:
        if req not in coord_names:
            msg = f"Unable to match any of the coordinates in dat for {req}"
            raise ValueError(msg)

    return coord_names


def check_var_SI_units(dat: DataArray, var: str,
                       enforce: bool = False) -> bool:

    if 'units' not in dat.attrs:
        msg = f'units is not an attribute of the given DataArray'
        raise ValueError(msg)

    units_SI = dat.units in var_units[var]
    if (enforce is True) and (units_SI is False):
        msg = f"The units '{dat.units}' do not match SI units for the {var}"+\
              f" category."
        raise ValueError(msg)

    return units_SI
