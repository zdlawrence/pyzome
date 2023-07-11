from __future__ import annotations

import numpy as np
import xarray as xr

from .checks import infer_xr_coord_names, check_var_SI_units, check_for_logp_coord
from .constants import (
    PREF,
    RHOREF,
    SCALE_HEIGHT,
    GAS_CONST_DRY_AIR,
    SPEC_HEAT_DRY_AIR,
    EARTH_RADIUS,
    EARTH_ROTA_RATE,
)


def add_logp_altitude(
    dat: xr.Dataset | xr.DataArray,
    lev_coord: str = "",
    H: float = SCALE_HEIGHT,
    p0: float = PREF,
) -> xr.Dataset | xr.DataArray:
    r"""A convenience function for adding a log-pressure altitude coordinate
    to an xarray Dataset or DataArray

    Parameters
    ----------
    dat : `xarray.Dataset` or `xarray.DataArray`
        data containing a pressure coordinate in SI units (Pa)
    lev_coord : string, optional
        The name of the pressure coordinate in the input data. Defaults
        to an empty string, for which the function will try to infer the
        pressure coordinate
    H : float, optional
        Scale height used to compute the log-pressure altitude
    p0 : float, optional
        Reference pressure for computation of log-pressure altitude.
        Defaults to 100000 Pa for Earth.

    Returns
    -------
    `xarray.Dataset` or `xarray.DataArray`
        The input data with an added log-pressure altitude coordinate named 'z'

    """

    if lev_coord == "":
        coords = infer_xr_coord_names(dat, required=["lev"])
        lev_coord = coords["lev"]
    check_var_SI_units(dat[lev_coord], "pressure", enforce=True)

    z = -H * np.log(dat[lev_coord] / p0)
    z.attrs["units"] = "m"  # type: ignore
    z.attrs["long_name"] = "log-pressure altitude"  # type: ignore
    z.attrs["note"] = "added by pyzome"  # type: ignore

    return dat.assign_coords({"z": z})


def buoyancy_frequency_squared(
    T: xr.DataArray,
    Rs: float = GAS_CONST_DRY_AIR,
    Cp: float = SPEC_HEAT_DRY_AIR,
    H: float = SCALE_HEIGHT,
    p0: float = PREF,
) -> xr.DataArray:
    r"""Calculates the buoyancy frequency squared given temperature.

    Parameters
    ----------
    T : `xarray.DataArray`
        The temperature data in units of Kelvin
    Rs : float, optional
        Specific gas constant. Defaults to 287.058 J/kg/K for dry
        air of the Earth.
    Cp : float, optional
        Specific heat capacity at constant pressure. Defaults to
        1004.64 J/kg/K for dry air of the Earth.
    H : float, optional
        The scale height used to calculate the log-pressure altitude.
        Defaults to 7000 m.
    p0 : float, optional
        Reference pressure used to calculate the log-pressure altitude.
        Defaults to 100000 Pa for Earth.

    Returns
    -------
    Nsq: `xarray.DataArray`
        The buoyancy frequency squared, in units of s-2.

    """

    check_for_logp_coord(T, enforce=True)
    check_var_SI_units(T, "temperature", enforce=True)

    dT_dz = T.differentiate("z", edge_order=2)
    Nsq = (Rs / H) * (dT_dz + (Rs / Cp) * (T / H))
    Nsq.attrs["units"] = "s-2"
    Nsq.attrs["long_name"] = "buoyancy frequency squared"

    return Nsq


def merid_grad_qgpv(
    u: xr.DataArray,
    Nsq: xr.DataArray | float,
    lat_coord: str = "",
    rho_s: float = RHOREF,
    H: float = SCALE_HEIGHT,
    Omega: float = EARTH_ROTA_RATE,
    a: float = EARTH_RADIUS,
    terms: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    r"""Calculates the meridional gradient in the quasi-geostrophic
    potential vorticity (QGPV) given the zonal mean zonal winds, and
    the squared buoyancy frequency.

    Parameters
    ----------
    u : `xarray.DataArray`
        The zonal mean zonal wind data.
    Nsq : `xarray.DataArray` or float
        The squared buoyancy frequency. Nsq need not have the same dimensions
        as u, but it should be consistent with u in the sense that it must be
        able to be properly broadcasted when used in computations with u (it is
        common to use a reference Nsq that only varies with altitude, or
        latitude/altitude)
    lat_coord : str, optional
        The coordinate name of the latitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the latitude
    rho_s : float, optional
        The reference density for air at the surface. Defaults to 1.2 kg m-3
        for density of air on Earth.
    H : float, optional
        The scale height used to calculate the log-pressure altitude.
        Defaults to 7000 m.
    Omega : float, optional
        Planetary rotation rate. Defaults to 7.29211e-5 s-1 for the Earth.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.
    terms : bool, optional
        If False (the default), the function returns the sum of the 3 terms
        making up the QGPV. If True, the function returns the 3 individual
        terms as a tuple containing the contributions from (1) the change in
        planetary vorticity with latitude; (2) the horizontal curvature of the
        zonal mean zonal wind; and (3) the vertical curvature of the zonal mean
        zonal wind.

    Returns
    -------
    qgpv_grad: `xarray.DataArray` or tuple of DataArrays
        Depending on the value of the terms keyword argument, either the full
        meridional QGPV gradient field, or a tuple of the individual terms
        that make up the total, in units of s-1.

    Todo
    -----
    * Unit checks on u and Nsq (if Nsq is a DataArray)
    * Add units on output terms

    """

    check_for_logp_coord(u, enforce=True)
    # note that Nsq is *not* checked like u, since a user may want to
    # use a reference Nsq independent of latitude and/or time

    if lat_coord == "":
        coords = infer_xr_coord_names(u, required=["lat"])
        lat_coord = coords["lat"]

    r2d = 180.0 / np.pi
    phi = np.deg2rad(u[lat_coord])
    cosphi = np.cos(phi)
    f = 2 * Omega * np.sin(phi)

    # contribution from change in planetary vorticity
    grad_coriolis = 2 * Omega * cosphi

    # contribution from meridional wind curvature
    horiz_curv = (
        r2d * (u * cosphi).differentiate(lat_coord, edge_order=2) / (a * cosphi)
    )
    horiz_curv = -r2d * horiz_curv.differentiate(lat_coord, edge_order=2)

    # contribution from vertical wind curvature
    rho_0 = rho_s * np.exp(-u.z / H)
    du_dz = u.differentiate("z", edge_order=2)
    verti_curv = (-a * f * f / rho_0) * ((rho_0 / Nsq) * du_dz).differentiate(
        "z", edge_order=2
    )

    # return individual terms if requested, otherwise the sum
    if terms is True:
        return (grad_coriolis, horiz_curv, verti_curv)  # type: ignore
    else:
        return grad_coriolis + horiz_curv + verti_curv


def refractive_index(
    u: xr.DataArray,
    q_phi: xr.DataArray,
    Nsq: xr.DataArray | float,
    k: int,
    phase_speed: float = 0,
    lat_coord: str = "",
    rho_s: float = RHOREF,
    H: float = SCALE_HEIGHT,
    Omega: float = EARTH_ROTA_RATE,
    a: float = EARTH_RADIUS,
    cd_approx_term: bool = False,
    terms: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    r"""Calculates the (squared) refractive index for a given distribution of
    zonal mean zonal winds, meridional QGPV gradients, buoyancy frequency
    squared, and zonal wavenumber+phase speed.

    Parameters
    ----------
    u : `xarray.DataArray`
        The zonal mean zonal wind data.
    q_phi : `xarray.DataArray`
        The meridional QGPV gradient
    Nsq : `xarray.DataArray` or float
        The squared buoyancy frequency. Nsq need not have the same dimensions
        as u, but it should be consistent with u and q_phi in the sense that it
        must be able to be properly broadcasted when used in computations (it is
        common to use a reference Nsq that is constant value or time-mean field)
    k : int
        The zonal wavenumber
    phase_speed : float, optional
        The phase speed of the wave, in m s-1.
    lat_coord : str, optional
        The coordinate name of the latitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the latitude
    rho_s : float, optional
        The reference density for air at the surface. Defaults to 1.2 kg m-3
        for density of air on Earth.
    H : float, optional
        The scale height used to calculate the log-pressure altitude.
        Defaults to 7000 m.
    Omega : float, optional
        Planetary rotation rate. Defaults to 7.29211e-5 s-1 for the Earth.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.
    cd_approx_term : bool, optional
        If False (the default), the function uses the standard buoyancy
        frequency term involving only Nsq. If True, the function uses an
        approximation originally derived by Charney & Drazin involving
        the vertical curvature of the buoyancy frequency. See Weinberger
        et al., 2021, "The Efficiency of Upward Wave Propagation near the
        Tropopause: Importance of the Form of the Refractive Index" for
        details of this term.
    terms : bool, optional
        If False (the default), the function returns the sum of the 3 terms
        making up the squared refractive index. If True, the function returns
        the 3 individual terms as a tuple containing the contributions from
        (1) the contribution from the meridional QGPV gradient; (2) the zonal
        wavenumber; and (3) the buoyancy.

    Returns
    -------
    RIsq: `xarray.DataArray` or tuple of DataArrays
        Depending on the value of the terms keyword argument, either the full
        squared refractive index field, or a tuple of the individual terms
        that make the total, in units of m-2.

    Todo
    -----
    * Unit checks on u, Nsq (if Nsq is a DataArray), and q_phi
    * Add units on output terms

    """

    check_for_logp_coord(u, enforce=True)
    check_for_logp_coord(q_phi, enforce=True)
    if lat_coord == "":
        coords = infer_xr_coord_names(u, required=["lat"])
        lat_coord = coords["lat"]

    lats = np.deg2rad(u[lat_coord])
    cosphi = np.cos(lats)
    f = 2 * Omega * np.sin(lats)

    qgpv_grad_term = q_phi / (a * (u - phase_speed))
    wavenum_term = -k * k / (a * a * cosphi * cosphi)

    # See Weinberger et al., 2021 for reference
    if cd_approx_term is True:
        rho_0 = rho_s * np.exp(-u.z / H)
        d_dz = np.sqrt(rho_0 / Nsq).differentiate("z", edge_order=2)
        d2_dz2 = d_dz.differentiate("z", edge_order=2)
        buoyancy_term = -(f * f / np.sqrt(Nsq)) * (1 / np.sqrt(rho_0)) * d2_dz2
    else:
        buoyancy_term = -f * f / (4 * Nsq * H * H)

    if terms is True:
        return (qgpv_grad_term, wavenum_term, buoyancy_term)  # type: ignore
    else:
        return qgpv_grad_term + wavenum_term + buoyancy_term


def plumb_wave_activity_flux(
    psip: xr.DataArray,
    Nsq: xr.DataArray | float,
    components: list[str] = ["x", "y", "z"],
    lat_coord: str = "",
    lon_coord: str = "",
    Omega: float = EARTH_ROTA_RATE,
    a: float = EARTH_RADIUS,
) -> list[xr.DataArray]:
    r"""Calculates the components of the Plumb wave activity flux given the eddy
    streamfunction and buoyancy frequency squared.

    Parameters
    ----------
    psip : `xarray.DataArray`
        The eddy streamfunction in units of m+2 s-1
    Nsq : `xarray.DataArray` or float
        The squared buoyancy frequency. Nsq need not have the same dimensions
        as psip, but it should be consistent with psip in the sense that it
        must be able to be properly broadcasted when used in computations (it is
        common to use a reference Nsq that is constant value or time-mean field)
    components : list, optional
        The components of the wave activity flux to compute and return in the
        output. 'x' refers to the east-west component, 'y' refers to the north-
        south component, and 'z' refers to the vertical component.
    lat_coord : str, optional
        The coordinate name of the latitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the latitude
    lon_coord : str, optional
        The coordinate name of the longitude dimension. If given an empty string
        (the default), the function tries to infer which coordinate corresponds
        to the longitude
    Omega : float, optional
        Planetary rotation rate. Defaults to 7.29211e-5 s-1 for the Earth.
    a : float, optional
        Planetary radius. Defaults to 6.37123e6 m for the Earth.

    Returns
    -------
    waf: list of `xarray.DataArray`s
        The wave activity flux vector components consistent with the
        desired components keyword argument.

    Todo
    -----
    * Unit checks on psip and Nsq (if Nsq is a DataArray)
    * Add units on output terms

    """

    check_for_logp_coord(psip, enforce=True)
    coords = infer_xr_coord_names(psip, required=["lev"])

    if ("y" in components) and ("lat" not in coords):
        msg = "A latitude coordinate must be available"
        raise ValueError(msg)
    elif (("x" in components) or ("z" in components)) and ("lon" not in coords):
        msg = "A longitude coordinate must be available"
        raise ValueError(msg)

    lat_coord = coords["lat"]
    lon_coord = coords["lon"]
    r2d = 180.0 / np.pi
    lats = np.deg2rad(psip[lat_coord])
    cosphi = np.cos(lats)
    f = 2 * Omega * np.sin(lats)

    p = psip[coords["lev"]] / 100000.0

    waf = []
    dpsi_dlam = r2d * psip.differentiate(lon_coord, edge_order=2)
    if "x" in components:
        d2psi_dlam2 = r2d * dpsi_dlam.differentiate(lon_coord, edge_order=2)
        wafx = (p / (2 * a * a * cosphi)) * ((dpsi_dlam) ** 2 - psip * d2psi_dlam2)
        waf.append(wafx)
    if "y" in components:
        dpsi_dphi = r2d * psip.differentiate(lat_coord, edge_order=2)
        d2psi_dphidlam = r2d * dpsi_dphi.differentiate(lon_coord, edge_order=2)
        wafy = (p / (2 * a * a)) * (dpsi_dlam * dpsi_dphi - psip * d2psi_dphidlam)
        waf.append(wafy)
    if "z" in components:
        dpsi_dz = psip.differentiate("z", edge_order=2)
        d2psi_dlamdz = dpsi_dlam.differentiate("z", edge_order=2)
        wafz = (p * f * f / (2 * Nsq * a)) * (dpsi_dlam * dpsi_dz - psip * d2psi_dlamdz)
        waf.append(wafz)

    return waf
