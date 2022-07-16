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

    check_for_logp_coord(T, enforce=True)

    dT_dz = T.differentiate("z", edge_order=2)
    Nsq = (Rs/H)*(dT_dz + (Rs/Cp)*(T/H))
    return Nsq


def merid_grad_qgpv(u, Nsq, lat_coord="", rho_s=RHOREF, H=SCALE_HEIGHT,
                    Omega=EARTH_ROTA_RATE, a=EARTH_RADIUS, terms=False):

    check_for_logp_coord(u, enforce=True)
    # note that Nsq is *not* checked like u, since a user may want to
    # use a constant Nsq

    if lat_coord == "":
        coords = infer_xr_coord_names(u, required=["lat"])
        lat_coord = coords["lat"]

    lats = np.deg2rad(u[lat_coord])
    cosphi = np.cos(lats)
    f = 2*Omega*np.sin(lats)

    # contribution from change in planetary vorticity
    grad_coriolis = 2*Omega*cosphi

    # contribution from meridional wind curvature
    horiz_curv = (u*cosphi).differentiate(lat_coord, edge_order=2)/(a*cosphi)
    horiz_curv = -horiz_curv.differentiate(lat_coord, edge_order=2)

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
        buoyancy_term = -(f*f/Nsq)*(1/np.sqrt(rho_0))*d2_dz2
    else:
        buoyancy_term = -f*f / (4*Nsq*H*H)

    if (terms is True):
        return (qgpv_grad_term, wavenum_term, buoyancy_term)
    else:
        return qgpv_grad_term + wavenum_term + buoyancy_term
