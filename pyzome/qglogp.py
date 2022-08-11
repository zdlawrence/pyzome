import numpy as np
import xarray as xr

from .checks import (
    infer_xr_coord_names,
    check_var_SI_units,
    check_for_logp_coord
)
from .constants import (
    PREF, RHOREF, SCALE_HEIGHT,
    GAS_CONST_DRY_AIR, SPEC_HEAT_DRY_AIR,
    EARTH_RADIUS, EARTH_ROTA_RATE
)


def add_logp_altitude(dat, lev_coord="", H=SCALE_HEIGHT, p0=PREF):
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

    z = -H * np.log(dat[lev_coord]/p0)
    z.attrs["units"] = "m"
    z.attrs["long_name"] = "log-pressure altitude"

    return dat.assign_coords({"z": z})


def buoyancy_frequency_squared(T, Rs=GAS_CONST_DRY_AIR, Cp=SPEC_HEAT_DRY_AIR,
                               H=SCALE_HEIGHT, p0=PREF):
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
    `xarray.DataArray`
        The buoyancy frequency squared, in units of s-2.

    """

    check_for_logp_coord(T, enforce=True)
    check_var_SI_units(T, "temperature", enforce=True)

    dT_dz = T.differentiate("z", edge_order=2)
    Nsq = (Rs/H)*(dT_dz + (Rs/Cp)*(T/H))
    Nsq.attrs["units"] = "s-2"

    return Nsq


def merid_grad_qgpv(u, Nsq, lat_coord="", rho_s=RHOREF, H=SCALE_HEIGHT,
                    Omega=EARTH_ROTA_RATE, a=EARTH_RADIUS, terms=False):
    
    r"""Calculates the meridional gradient in the quasi-geostrophic 
    potential vorticity (QGPV) given the zonal mean zonal winds, and 
    the squared buoyancy frequency.

    Parameters
    ----------
    u : `xarray.DataArray`
        The zonal mean zonal wind data.
    Nsq : `xarray.DataArray`
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
        terms as a tuple containing the contributions from (1) the 
        change in planetary vorticity with latitude; (2) the horizontal 
        curvature of the zonal mean zonal wind; and (3) the vertical 
        curvature of the zonal mean zonal wind. 

    Returns
    -------
    `xarray.DataArray` or tuple of DataArrays
        Depending on the value of the terms keyword argument, either the 
        full meridional QGPV gradient field, or the individual terms 
        making it up, in units of s-1.

    """

    check_for_logp_coord(u, enforce=True)
    # note that Nsq is *not* checked like u, since a user may want to
    # use a reference Nsq independent of latitude and/or time

    if lat_coord == "":
        coords = infer_xr_coord_names(u, required=["lat"])
        lat_coord = coords["lat"]

    r2d = 180./np.pi
    phi = np.deg2rad(u[lat_coord])
    cosphi = np.cos(phi)
    f = 2*Omega*np.sin(phi)

    # contribution from change in planetary vorticity
    grad_coriolis = 2*Omega*cosphi

    # contribution from meridional wind curvature
    horiz_curv = r2d*(u*cosphi).differentiate(lat_coord, edge_order=2)/(a*cosphi)
    horiz_curv = -r2d*horiz_curv.differentiate(lat_coord, edge_order=2)

    # contribution from vertical wind curvature
    rho_0 = rho_s * np.exp(-u.z / H)
    du_dz = u.differentiate("z", edge_order=2)
    verti_curv = (-a*f*f/rho_0)*((rho_0/Nsq)*du_dz).differentiate("z", edge_order=2)

    # return individual terms if requested, otherwise the sum
    if (terms is True):
        return (grad_coriolis, horiz_curv, verti_curv)
    else:
        return grad_coriolis + horiz_curv + verti_curv


def refractive_index(u, q_phi, Nsq, k, phase_speed=0, lat_coord="",
                     rho_s=RHOREF, H=SCALE_HEIGHT,
                     Omega=EARTH_ROTA_RATE, a=EARTH_RADIUS,
                     cd_approx_term=False, terms=False):
    r"""Calculates the (squared) refractive index for a given distribution 
    of zonal mean zonal winds, meridional QGPV gradients, buoyancy frequency 
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
        frequency term involving only Nsq. If True, the function uses 
        an approximation originally derived by Charney & Drazin involving 
        the vertical curvature of the buoyancy frequency. See Weinberger 
        et al., 2021, "The Efficiency of Upward Wave Propagation near the 
        Tropopause: Importance of the Form of the Refractive Index" for
        details of this term.
    terms : bool, optional
        If False (the default), the function returns the sum of the 3 terms
        making up the squared refractive index. If True, the function returns 
        the 3 individual terms as a tuple containing the contributions from 
        (1) the contribution from the meridional QGPV gradient; (2) the 
        zonal wavenumber; and (3) the buoyancy.

    Returns
    -------
    `xarray.DataArray` or tuple of DataArrays
        Depending on the value of the terms keyword argument, either the 
        full refractive index (squared) field, or the individual terms 
        making it up, in units of m-2.

    """

    check_for_logp_coord(u, enforce=True)
    check_for_logp_coord(q_phi, enforce=True)
    if lat_coord == "":
        coords = infer_xr_coord_names(u, required=["lat"])
        lat_coord = coords["lat"]

    lats = np.deg2rad(u[lat_coord])
    cosphi = np.cos(lats)
    f = 2*Omega*np.sin(lats)

    qgpv_grad_term = q_phi/(a*(u - phase_speed))
    wavenum_term = -k*k/(a*a*cosphi*cosphi)

    # See Weinberger et al., 2021 for reference
    if (cd_approx_term is True):
        rho_0 = rho_s * np.exp(-u.z / H)
        d_dz = np.sqrt(rho_0/Nsq).differentiate("z", edge_order=2)
        d2_dz2 = d_dz.differentiate("z", edge_order=2)
        buoyancy_term = -(f*f/np.sqrt(Nsq))*(1/np.sqrt(rho_0))*d2_dz2
    else:
        buoyancy_term = -f*f / (4*Nsq*H*H)

    if (terms is True):
        return (qgpv_grad_term, wavenum_term, buoyancy_term)
    else:
        return qgpv_grad_term + wavenum_term + buoyancy_term
