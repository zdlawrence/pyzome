import numpy as np

from .basic import zonal_mean
from .constants import PREF, GAS_CONST_DRY_AIR, SPEC_HEAT_DRY_AIR, EARTH_RADIUS, EARTH_ROTA_RATE


def resid_vel_ff(v, w, T, p0=PREF, Rs=GAS_CONST_DRY_AIR,
                 Cp=SPEC_HEAT_DRY_AIR, a=EARTH_RADIUS):
    r"""Calculate the residual mean velocity components using "full" fields.
    Full implies that the fields contain both longitude and latitude dimensions.

    Parameters
    ----------
    v : `xarray.DataArray`
        data containing the full field meridional wind component
    w : `xarray.DataArray`
        data containing the full field vertical *pressure* velocity component
    T : `xarray.DataArray`
        data containing the full field air temperature
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (v_res, w_res), the meridional and vertical
        residual velocity components.

    See Also
    --------
    resid_vel

    Notes
    -----
    v, w, and T should have the same dimensions

    """

    vzm = zonal_mean(v)
    wzm = zonal_mean(w)
    Tzm = zonal_mean(T)
    vT = zonal_mean((v-vzm) * (T-Tzm))

    return resid_vel(vzm, wzm, Tzm, vT, p0=p0, Rs=Rs, Cp=Cp, a=a)


def resid_vel(v, w, T, vT, p0=PREF, Rs=GAS_CONST_DRY_AIR,
              Cp=SPEC_HEAT_DRY_AIR, a=EARTH_RADIUS):
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (v_res, w_res), the meridional and vertical
        residual velocity components.

    See Also
    --------
    resid_vel_ff

    Notes
    -----
    v, w, T, and vT should generally have the same
    dimensions. However, as a consequence of the way xarray
    performs broadcasting of arrays, this function will still
    work *as long as all the arrays have at least latitude and
    level dimensions*. This has an added benefit that if you
    desire to compute the contribution of different zonal
    wavenumbers to the residual circulation, then you can provide
    the zonal covariances (vT) with an added dimension
    such as "lon_wavenum" (as is returned by the
    zonal_wave_covariance function) to get the correct result.


    TO DO
    -----
    * Do not assume the 'latitude' or 'level' dimension names
    * Make calculations unit-aware
    * Do not assume vertical pressure velocity

    """

    cos_lats = np.cos(np.deg2rad(v.latitude))
    to_theta = (p0/v.level)**(Rs/Cp)

    Tht = T * to_theta
    vTht = vT * to_theta

    dTht_dp = Tht.differentiate('level', edge_order=2)

    v_res = v - (vTht / dTht_dp).differentiate('level', edge_order=2)

    w_part = (vTht * cos_lats) / dTht_dp
    w_part = (180./np.pi)*w_part.differentiate('latitude', edge_order=2)
    w_res = w + (1 / (a*cos_lats)) * w_part

    return (v_res, w_res)


def epflux_vector_ff(u, v, w, T, p0=PREF, Rs=GAS_CONST_DRY_AIR,
                     Cp=SPEC_HEAT_DRY_AIR, a=EARTH_RADIUS, Omega=EARTH_ROTA_RATE):
    r"""Calculate the Eliassen-Palm Flux Vector components using full fields.
    Full implies that the input fields contain both longitude and latitude
    dimensions.

    Parameters
    ----------
    u : `xarray.DataArray`
        data containing the full field zonal wind component
    v : `xarray.DataArray`
        data containing the full field meridional wind component
    w : `xarray.DataArray`
        data containing the full field vertical *pressure* velocity component
    T : `xarray.DataArray`
        data containing the full field air temperature
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.
    Omega : float, optional
        Rotation rate of planetary body. Defaults to 7.29211e-5 for the Earth.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (F_lat, F_prs), the meridional and vertical
        components of the EP-Flux.

    See Also
    --------
    epflux_vector

    Notes
    -----
    u, v, w, and T should have the same dimensions.

    """

    # Take zonal means
    uzm = zonal_mean(u)
    vzm = zonal_mean(v)
    wzm = zonal_mean(w)
    Tzm = zonal_mean(T)

    # Get eddies (remove zonal mean)
    ued = u-uzm
    ved = v-vzm
    wed = w-wzm
    Ted = T-Tzm

    # Get necessary zonal covariances
    uv = zonal_mean(ued*ved)
    vT = zonal_mean(ved*Ted)
    uw = zonal_mean(ued*wed)

    return epflux_vector(u, T, uv, vT, uw, p0=p0, Rs=Rs, Cp=Cp, a=a, Omega=Omega)


def epflux_vector(u, T, uv, vT, uw, p0=PREF, Rs=GAS_CONST_DRY_AIR,
                  Cp=SPEC_HEAT_DRY_AIR, a=EARTH_RADIUS, Omega=EARTH_ROTA_RATE):
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.
    Omega : float, optional
        Rotation rate of planetary body. Defaults to 7.29211e-5 for the Earth.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (F_lat, F_prs), the meridional and vertical
        components of the EP-Flux.

    See Also
    --------
    epflux_vector_ff

    Notes
    -----
    u, T, uv, vT, and uw should generally have the same
    dimensions. However, as a consequence of the way xarray
    performs broadcasting of arrays, this function will still
    work *as long as all the arrays have at least latitude and
    level dimensions*. This has an added benefit that if you
    desire to compute EP Fluxes partitioned into contributions
    from different zonal wavenumbers, then you can provide the
    zonal covariances (uv, vT, and uw) with an added dimension
    such as "lon_wavenum" (as is returned by the
    zonal_wave_covariance function) to get the correct result.

    TO DO
    -----
    * Do not assume the 'latitude' or 'level' dimension names
    * Make calculations unit-aware
    * Do not assume vertical pressure velocity

    """

    f = 2*Omega*np.sin(np.deg2rad(u.latitude))
    cos_lats = np.cos(np.deg2rad(u.latitude))
    to_theta = (p0/u.level)**(Rs/Cp)

    Tht = T * to_theta
    vTht = vT * to_theta

    dTht_dp = Tht.differentiate('level', edge_order=2)
    du_dp = u.differentiate('level', edge_order=2)
    ducos_dphi = (180./np.pi)*(u*cos_lats).differentiate('latitude', edge_order=2)

    F_lat = a*cos_lats * ((vTht / dTht_dp)*du_dp - uv)
    F_prs = a*cos_lats * (((-vTht/dTht_dp)*(1/(a*cos_lats))*ducos_dphi) + (f*vTht/dTht_dp) - uw)

    return (F_lat, F_prs)


def qg_epflux_vector_ff(u, v, T, p0=PREF, Rs=GAS_CONST_DRY_AIR,
                        Cp=SPEC_HEAT_DRY_AIR, a=EARTH_RADIUS, Omega=EARTH_ROTA_RATE):
    r"""Calculate the quasi-geostrophic Eliassen-Palm Flux Vector
    components using full fields. Full implies that the input fields
    contain both longitude and latitude dimensions.

    Parameters
    ----------
    u : `xarray.DataArray`
        data containing the full field zonal wind component
    v : `xarray.DataArray`
        data containing the full field meridional wind component
    T : `xarray.DataArray`
        data containing the full field air temperature
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.
    Omega : float, optional
        Rotation rate of planetary body. Defaults to 7.29211e-5 for the Earth.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (F_lat, F_prs), the meridional and vertical
        components of the quasi-geostrophic EP-Flux.

    See Also
    --------
    qg_epflux_vector

    Notes
    -----
    u, v, and T should have the same dimensions.

    """

    # Take zonal means
    uzm = zonal_mean(u)
    vzm = zonal_mean(v)
    Tzm = zonal_mean(T)

    # Get eddies (remove zonal mean)
    ued = u-uzm
    ved = v-vzm
    Ted = T-Tzm

    # Get necessary zonal covariances
    uv = zonal_mean(ued*ved)
    vT = zonal_mean(ved*Ted)

    return qg_epflux_vector(T, uv, vT, p0=p0, Rs=Rs, Cp=Cp, a=a, Omega=Omega)


def qg_epflux_vector(T, uv, vT, p0=PREF, Rs=GAS_CONST_DRY_AIR,
                     Cp=SPEC_HEAT_DRY_AIR, a=EARTH_RADIUS, Omega=EARTH_ROTA_RATE):
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.
    Omega : float, optional
        Rotation rate of planetary body. Defaults to 7.29211e-5 for the Earth.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (F_lat, F_prs), the meridional and vertical
        components of the quasi-geostrophic EP-Flux.

    See Also
    --------
    qg_epflux_vector_ff

    Notes
    -----
    T, uv, and vT should generally have the same dimensions.
    However, as a consequence of the way xarray
    performs broadcasting of arrays, this function will still
    work *as long as all the arrays have at least latitude and
    level dimensions*. This has an added benefit that if you
    desire to compute QG-EP Fluxes partitioned into contributions
    from different zonal wavenumbers, then you can provide the
    zonal covariances (uv and vT) with an added dimension
    such as "lon_wavenum" (as is returned by the
    zonal_wave_covariance function) to get the correct result.


    TO DO
    -----
    * Do not assume the 'latitude' or 'level' dimension names

    """

    f = 2*Omega*np.sin(np.deg2rad(T.latitude))
    cos_lats = np.cos(np.deg2rad(T.latitude))
    to_theta = (p0/T.level)**(Rs/Cp)

    Tht = T * to_theta
    vTht = vT * to_theta

    dTht_dp = Tht.differentiate('level', edge_order=2)

    F_lat = -a*cos_lats*uv
    F_prs = a*cos_lats*f*(vTht/dTht_dp)

    return (F_lat, F_prs)


def epflux_div(F_lat, F_prs, accel=False, terms=False, a=EARTH_RADIUS):
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
        Radius of planetary sphere. Defaults to 6.37123e6 m for the Earth.

    Returns
    -------
    `xarray.DataArray` or tuple of (`xarray.DataArray`, `xarray.DataArray`)
        The total EP-Flux divergence (if terms=False) or
        merid_div and verti_div (if terms=True)

    See Also
    --------
    epflux_vector_ff
    epflux_vector
    qg_epflux_vector_ff
    qg_epflux

    TO DO
    -----
    * Do not assume the 'latitude' or 'level' dimension names

    """
    cos_lats = np.cos(np.deg2rad(F_lat.latitude))
    scale = (1/(a*cos_lats))

    merid_div = (180./np.pi)*scale*(F_lat*cos_lats).differentiate('latitude', edge_order=2)
    verti_div = F_prs.differentiate('level', edge_order=2)

    if (accel is True):
        merid_div *= scale
        verti_div *= scale

    if (terms is True):
        return (merid_div, verti_div)
    else:
        return merid_div+verti_div
