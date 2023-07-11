from __future__ import annotations

import numpy as np
import xarray as xr

from .checks import infer_xr_coord_names, check_var_SI_units
from .constants import (
    PREF,
    GAS_CONST_DRY_AIR,
    SPEC_HEAT_DRY_AIR,
    EARTH_RADIUS,
    EARTH_ROTA_RATE,
)


def resid_vel(
    v: xr.DataArray,
    w: xr.DataArray,
    T: xr.DataArray,
    vT: xr.DataArray,
    p0: float = PREF,
    Rs: float = GAS_CONST_DRY_AIR,
    Cp: float = SPEC_HEAT_DRY_AIR,
    a: float = EARTH_RADIUS,
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Calculate the residual mean velocity components using zonal mean fields.

    Parameters
    ----------
    v : `xarray.DataArray`
        data containing the zonal mean of the meridional wind
    w : `xarray.DataArray`
        data containing the zonal mean of the vertical *pressure* velocity
    T : `xarray.DataArray`
        data containing the zonal mean of the air temperature
    vT : `xarray.DataArray`
        data containing the zonal mean meridional eddy heat flux, nominally
        defined as zonal_mean((v - zonal_mean(v))*(T - zonal_mean(T)))
    lat_coord : str, optional
        The coordinate name of the latitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the latitude
    lev_coord : str, optional
        The coordinate of the pressure-level dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the pressure-levels
    p0 : float, optional
        Reference pressure for computation of potential temperature. Defaults
        to 100000 Pa for Earth.
    Rs : float, optional
        Specific gas constant for computation of potential temperature. Defaults
        to 287.058 J/kg/K for dry air of the Earth.
    Cp : float, optional
        Specific heat capacity at constant pressure for computation of potential
        temperature. Defaults to 1004.64 J/kg/K for dry air of the Earth.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.

    Returns
    -------
    residual velocities: tuple of two `xarray.DataArray`
        The meridional and vertical residual velocity components.

    Notes
    -----
    v, w, T, and vT should generally have the same dimensions. However, as
    a consequence of the way xarray performs broadcasting of arrays, this
    function will still work *as long as all the arrays have at least latitude
    and level dimensions*. This has an added benefit that if you desire to
    compute the contribution of different zonal wavenumbers to the residual
    circulation, then you can provide the zonal covariances (vT) with an added
    dimension such as "lon_wavenum" (as is returned by the zonal_wave_covariance
    function) to get the correct result.

    """

    coords = infer_xr_coord_names(v, required=["lat", "lev"])
    lat = coords["lat"]
    lev = coords["lev"]

    check_var_SI_units(v[lev], "pressure", enforce=True)
    check_var_SI_units(v, "wind", enforce=True)
    check_var_SI_units(w, "prs_vvel", enforce=True)
    check_var_SI_units(T, "temperature", enforce=True)
    check_var_SI_units(vT, "heatflux", enforce=True)

    cos_lats = np.cos(np.deg2rad(v[lat]))
    to_theta = (p0 / v[lev]) ** (Rs / Cp)

    Tht = T * to_theta
    vTht = vT * to_theta

    dTht_dp = Tht.differentiate(lev, edge_order=2)
    w_part = (vTht * cos_lats) / dTht_dp
    w_part = (180.0 / np.pi) * w_part.differentiate(lat, edge_order=2)

    v_res = v - (vTht / dTht_dp).differentiate(lev, edge_order=2)
    w_res = w + (1 / (a * cos_lats)) * w_part

    v_res.attrs["units"] = "m s-1"
    w_res.attrs["units"] = "Pa s-1"

    return (v_res, w_res)


def epflux_vector(
    u: xr.DataArray,
    T: xr.DataArray,
    uv: xr.DataArray,
    vT: xr.DataArray,
    uw: xr.DataArray,
    p0: float = PREF,
    Rs: float = GAS_CONST_DRY_AIR,
    Cp: float = SPEC_HEAT_DRY_AIR,
    a: float = EARTH_RADIUS,
    Omega: float = EARTH_ROTA_RATE,
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Calculate the Eliassen-Palm Flux Vector components using zonal mean fields.

    Parameters
    ----------
    u : `xarray.DataArray`
        data containing the zonal mean of the zonal wind
    T : `xarray.DataArray`
        data containing the zonal mean of the air temperature
    uv : `xarray.DataArray`
        data containing the zonal mean meridional momentum flux
    vT : `xarray.DataArray`
        data containing the zonal mean meridional heat flux
    uw : `xarray.DataArray`
        data containing the zonal mean vertical momentum flux
        (consistent with w, the vertical pressure velocity field)
    p0 : float, optional
        Reference pressure for computation of potential temperature. Defaults
        to 100000 Pa for Earth.
    Rs : float, optional
        Specific gas constant for computation of potential temperature. Defaults
        to 287.058 J/kg/K for dry air of the Earth.
    Cp : float, optional
        Specific heat capacity at constant pressure for computation of potential
        temperature. Defaults to 1004.64 J/kg/K for dry air of the Earth.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.
    Omega : float, optional
        Planetary rotation rate. Defaults to 7.29211e-5 s-1 for the Earth.

    Returns
    -------
    ep_flux: tuple of two `xarray.DataArray`
        The meridional and vertical components of the EP-Flux (F_lat, F_prs)

    Notes
    -----
    u, T, uv, vT, and uw should generally have the same dimensions. However,
    as a consequence of the way xarray performs broadcasting of arrays, this
    function will still work *as long as all the arrays have at least latitude
    and level dimensions*. This has an added benefit that if you desire to
    compute EP Fluxes partitioned into contributions from different zonal
    wavenumbers, then you can provide the zonal covariances (uv, vT, and uw)
    with an added dimension such as "zonal_wavenum" (as is returned by the
    zonal_wave_covariance function) to get the correct result.

    """

    coords = infer_xr_coord_names(u, required=["lat", "lev"])
    lat = coords["lat"]
    lev = coords["lev"]

    check_var_SI_units(u[lev], "pressure", enforce=True)
    check_var_SI_units(T, "temperature", enforce=True)
    check_var_SI_units(uv, "momentumflux", enforce=True)
    check_var_SI_units(uw, "prs_vertmomflux", enforce=True)
    check_var_SI_units(vT, "heatflux", enforce=True)

    f = 2 * Omega * np.sin(np.deg2rad(u[lat]))
    cos_lats = np.cos(np.deg2rad(u[lat]))
    to_theta = (p0 / u[lev]) ** (Rs / Cp)

    Tht = T * to_theta
    vTht = vT * to_theta

    dTht_dp = Tht.differentiate(lev, edge_order=2)
    du_dp = u.differentiate(lev, edge_order=2)
    ducos_dphi = (180.0 / np.pi) * (u * cos_lats).differentiate(lat, edge_order=2)

    F_lat = a * cos_lats * ((vTht / dTht_dp) * du_dp - uv)
    F_prs = (
        a
        * cos_lats
        * (
            ((-vTht / dTht_dp) * (1 / (a * cos_lats)) * ducos_dphi)
            + (f * vTht / dTht_dp)
            - uw
        )
    )

    F_lat.attrs["units"] = "m+3 s-2"  # type: ignore
    F_prs.attrs["units"] = "Pa m+2 s-2"  # type: ignore

    return (F_lat, F_prs)  # type: ignore


def qg_epflux_vector(
    T: xr.DataArray,
    uv: xr.DataArray,
    vT: xr.DataArray,
    p0: float = PREF,
    Rs: float = GAS_CONST_DRY_AIR,
    Cp: float = SPEC_HEAT_DRY_AIR,
    a: float = EARTH_RADIUS,
    Omega: float = EARTH_ROTA_RATE,
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Calculate the quasi-geostrophic Eliassen-Palm Flux Vector
    components using zonal mean fields.

    Parameters
    ----------
    T : `xarray.DataArray`
        data containing the full field air temperature
    uv : `xarray.DataArray`
        data containing the zonal mean meridional momentum flux
    vT : `xarray.DataArray`
        data containing the zonal mean meridional heat flux
    p0 : float, optional
        Reference pressure for computation of potential temperature. Defaults
        to 100000 Pa for Earth.
    Rs : float, optional
        Specific gas constant for computation of potential temperature. Defaults
        to 287.058 J/kg/K for dry air of the Earth.
    Cp : float, optional
        Specific heat capacity at constant pressure for computation of potential
        temperature. Defaults to 1004.64 J/kg/K for dry air of the Earth.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.
    Omega : float, optional
        Planetary rotation rate. Defaults to 7.29211e-5 s-1 for the Earth.

    Returns
    -------
    qg_ep_flux: tuple of two `xarray.DataArray`
        The meridional and vertical components of the quasi-geostrophic EP-Flux
        (F_lat, F_prs).

    Notes
    -----
    T, uv, and vT should generally have the same dimensions. However, as a
    consequence of the way xarray performs broadcasting of arrays, this function
    will still work *as long as all the arrays have at least latitude and level
    dimensions*. This has an added benefit that if you desire to compute QG-EP
    Fluxes partitioned into contributions from different zonal wavenumbers, then
    you can provide the zonal covariances (uv and vT) with an added dimension
    such as "zonal_wavenum" (as is returned by the zonal_wave_covariance function)
    to get the correct result.

    """

    coords = infer_xr_coord_names(T, required=["lat", "lev"])
    lat = coords["lat"]
    lev = coords["lev"]

    check_var_SI_units(T[lev], "pressure", enforce=True)
    check_var_SI_units(T, "temperature", enforce=True)
    check_var_SI_units(uv, "momentumflux", enforce=True)
    check_var_SI_units(vT, "heatflux", enforce=True)

    f = 2 * Omega * np.sin(np.deg2rad(T[lat]))
    cos_lats = np.cos(np.deg2rad(T[lat]))
    to_theta = (p0 / T[lev]) ** (Rs / Cp)

    Tht = T * to_theta
    vTht = vT * to_theta

    dTht_dp = Tht.differentiate(lev, edge_order=2)

    F_lat = -a * cos_lats * uv
    F_prs = a * cos_lats * f * (vTht / dTht_dp)

    F_lat.attrs["units"] = "m+3 s-2"  # type: ignore
    F_prs.attrs["units"] = "Pa m+2 s-2"  # type: ignore

    return (F_lat, F_prs)  # type: ignore


def epflux_div(
    F_lat: xr.DataArray,
    F_prs: xr.DataArray,
    accel: bool = False,
    terms: bool = False,
    a: float = EARTH_RADIUS,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    r"""Calculate the Eliassen-Palm Flux divergence assuming
    spherical coordinates.

    Parameters
    ----------
    F_lat : `xarray.DataArray`
        data containing the meridional EP-Flux component
    F_prs : `xarray.DataArray`
        data containing the vertical EP-Flux component
    accel : bool, optional
        If True, will scale the output by 1 / (a*cos(lat)) so
        that the divergence is in units of m s-2
    terms : bool, optional
        If True, the function returns the individual contributions
        to the divergence from the meridional and vertical components
        of the EP-Flux. Defaults to False - the function returns the
        sum of these terms.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.

    Returns
    -------
    epflux_divergence: `xarray.DataArray` or tuple of (`xarray.DataArray`, `xarray.DataArray`)
        The total EP-Flux divergence or
        the individual terms from the meridional and vertical divergence (if terms=True)

    See Also
    --------
    epflux_vector
    qg_epflux_vector

    Todo
    -----
    * Add unit checks on input EP-Flux components

    """

    coords = infer_xr_coord_names(F_lat, required=["lat", "lev"])
    lat = coords["lat"]
    lev = coords["lev"]

    check_var_SI_units(F_prs[lev], "pressure", enforce=True)

    cos_lats = np.cos(np.deg2rad(F_lat[lat]))
    scale = 1 / (a * cos_lats)

    merid_div = (
        (180.0 / np.pi) * scale * (F_lat * cos_lats).differentiate(lat, edge_order=2)
    )
    verti_div = F_prs.differentiate(lev, edge_order=2)

    if accel is True:
        merid_div *= scale
        merid_div.attrs["units"] = "m s-2"  # type: ignore
        verti_div *= scale
        verti_div.attrs["units"] = "m s-2"  # type: ignore
    else:
        merid_div.attrs["units"] = "m+2 s-2"  # type: ignore
        verti_div.attrs["units"] = "m+2 s-2"  # type: ignore

    if terms is True:
        return (merid_div, verti_div)  # type: ignore

    with xr.set_options(keep_attrs=True):
        return merid_div + verti_div  # type: ignore
