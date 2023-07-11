import pytest

import numpy as np

from pyzome.tem import (
    resid_vel,
    epflux_vector,
    qg_epflux_vector,
    epflux_div,
)
from pyzome.mock_data import create_dummy_geo_dataset, lat_coord, plev_coord
from pyzome.constants import EARTH_RADIUS


def test_resid_vel():
    """Test that resid_vel works correctly"""
    ds = create_dummy_geo_dataset(
        ["v", "T", "omega", "vT"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={
            "v": {"units": "m/s"},
            "omega": {"units": "Pa/s"},
            "T": {"units": "K"},
            "vT": {"units": "K m s-1"},
        },
    )

    v_star, w_star = resid_vel(ds.v, ds.omega, ds.T, ds.vT)
    assert v_star.attrs["units"] == "m s-1"
    assert w_star.attrs["units"] == "Pa s-1"


def test_epflux_vector():
    ds = create_dummy_geo_dataset(
        ["u", "T", "uv", "vT", "uw"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={
            "u": {"units": "m/s"},
            "T": {"units": "K"},
            "uv": {"units": "m+2 s-2"},
            "vT": {"units": "K m s-1"},
            "uw": {"units": "m Pa s-2"},
        },
    )

    F_lat, F_prs = epflux_vector(ds.u, ds.T, ds.uv, ds.vT, ds.uw)
    assert F_lat.attrs["units"] == "m+3 s-2"
    assert F_prs.attrs["units"] == "Pa m+2 s-2"


def test_qg_epflux_vector():
    ds = create_dummy_geo_dataset(
        ["T", "uv", "vT"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={
            "T": {"units": "K"},
            "uv": {"units": "m+2 s-2"},
            "vT": {"units": "K m s-1"},
        },
    )

    F_lat, F_prs = qg_epflux_vector(ds.T, ds.uv, ds.vT)
    assert F_lat.attrs["units"] == "m+3 s-2"
    assert F_prs.attrs["units"] == "Pa m+2 s-2"


def test_epflux_div():
    ds = create_dummy_geo_dataset(
        ["F_lat", "F_prs"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(5, 2, units="Pa"),
        field_attrs={
            "F_lat": {"units": "m+3 s-2"},
            "F_prs": {"units": "Pa m+2 s-2"},
        },
    )

    div = epflux_div(ds.F_lat, ds.F_prs)
    assert div.attrs["units"] == "m+2 s-2"

    cos_lats = np.cos(np.deg2rad(ds.lat))
    scale = 1 / (EARTH_RADIUS * cos_lats)
    div_accel = epflux_div(ds.F_lat, ds.F_prs, accel=True)
    assert div_accel.attrs["units"] == "m s-2"
    np.testing.assert_allclose((div * scale).values, div_accel.values)

    merid_div, verti_div = epflux_div(ds.F_lat, ds.F_prs, terms=True)
    assert merid_div.attrs["units"] == div.attrs["units"]
    assert verti_div.attrs["units"] == div.attrs["units"]
