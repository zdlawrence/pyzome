from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import xarray as xr
import scipy
import xrft

from .checks import has_global_regular_lons, infer_xr_coord_names
from .mock_data import lon_coord


def zonal_wave_coeffs(
    dat: xr.DataArray,
    *,
    waves: None | Iterable[int] = None,
    fftpkg: str = "scipy",
    lon_coord: str = "",
) -> xr.DataArray:
    r"""Compute the Fourier transform of data along the zonal direction.

    This function is imported at the top level of the package by default.

    Parameters
    ----------
    dat : `xarray.DataArray`
        input data containing a longitude dimension that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None
        for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are
        memory-eager and leverages scipy.fft.rfft. Specifying xrft should
        leverage the benefits of xarray/dask for large datasets by using
        xrft.fft. Defaults to scipy.
    lon_coord : str, optional
        The coordinate name of the longitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the longitude

    Returns
    -------
    `xarray.DataArray`
        Output of an rFFT along the longitude dimension, for specified
        waves only. The output is a complex-valued DataArray containing
        at least the `zonal_wavenum` dimension, with attributes `nlons`
        and `lon0` that specify the number of longitudes and the starting
        longitude of the input data.

    """

    if fftpkg not in ["scipy", "xrft"]:
        msg = "fftpkg must be 'scipy' or 'xrft'"
        raise ValueError(msg)

    if lon_coord == "":
        coords = infer_xr_coord_names(dat, required=["lon"])
        lon_coord = coords["lon"]
    has_global_regular_lons(dat[lon_coord], enforce=True)

    funcs = {"scipy": _zonal_wave_coeffs_scipy, "xrft": _zonal_wave_coeffs_xrft}
    fc = funcs[fftpkg](dat, lon_coord)

    fc.attrs["nlons"] = dat[lon_coord].size
    fc.attrs["lon0"] = dat[lon_coord].values[0]
    if waves is not None:
        fc = fc.sel(zonal_wavenum=waves)

    return fc


def _zonal_wave_coeffs_scipy(dat: xr.DataArray, lon_coord: str) -> xr.DataArray:
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    Uses scipy.fft.rfft to perform the calculation.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    lon_coord : string
        name of the dimension/coordinate corresponding to the longitudes

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.

    """
    nlons = dat[lon_coord].size
    lon_ax = dat.get_axis_num(lon_coord)

    new_dims = list(dat.dims)
    new_dims[lon_ax] = "zonal_wavenum"  # type: ignore

    new_coords = dict(dat.coords)
    new_coords.pop(lon_coord)
    new_coords["zonal_wavenum"] = np.arange(0, nlons // 2 + 1)

    fc = scipy.fft.rfft(dat.values, axis=lon_ax)
    fc = xr.DataArray(fc, coords=new_coords, dims=new_dims)

    return fc


def _zonal_wave_coeffs_xrft(dat: xr.DataArray, lon_coord: str) -> xr.DataArray:
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    Uses xrft.fft to perform the calculation.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    lon_coord : string
        name of the dimension/coordinate corresponding to the longitudes

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.

    """

    fc = xrft.fft(
        dat, dim=lon_coord, real_dim=lon_coord, true_phase=False, true_amplitude=False
    )

    fc = fc.rename({f"freq_{lon_coord}": "zonal_wavenum"})
    fc = fc.assign_coords({"zonal_wavenum": np.arange(fc.zonal_wavenum.size)})

    return fc


def inflate_zonal_wave_coeffs(
    fc_subset: xr.DataArray,
    wave_coord: str = "",
) -> xr.DataArray:
    r"""Inflates an array of zonal wavenumber Fourier coefficients to
    its expected full-spectrum size. The full size is determined by
    the "nlons" attribute of the input DataArray.

    Inflating is done by inserting zeros in the missing wavenumber bins.
    This can be useful for keeping only a subset of wavenumbers while still
    allowing for taking an inverse FFT for filtering or determining the
    contributions of individual wavenumbers to the full field.

    Parameters
    ----------
    fc_subset : `xarray.DataArray`
        Fourier coefficients as a function of zonal wavenumber, as
        returned by `zonal_wave_coeffs`.
    wave_coord : str, optional
        The coordinate name of the wavenumber dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the wavenumbers

    Returns
    -------
    `xarray.DataArray`
        Fourier coefficients as a function of zonal wavenumber, with the
        spectrum size determined by the "nlons" attribute of the input
        DataArray.

    """
    if wave_coord == "":
        coords = infer_xr_coord_names(fc_subset, required=["zonal_wavenum"])
        wave_coord = coords["zonal_wavenum"]

    if "nlons" not in fc_subset.attrs:
        msg = "input DataArray must have an 'nlons' attribute specifying the number of longitudes in the source data"
        raise KeyError(msg)

    expected_wavenums = np.arange(fc_subset.attrs["nlons"] // 2 + 1)
    if np.array_equiv(fc_subset[wave_coord], expected_wavenums):
        return fc_subset
    elif np.any(~np.in1d(fc_subset[wave_coord], expected_wavenums, assume_unique=True)):
        msg = "input DataArray wavenumbers are not a subset of the expected wavenumbers based on the 'nlons' attribute"
        raise ValueError(msg)

    return fc_subset.reindex({wave_coord: expected_wavenums}, fill_value=0j)  # type: ignore


def zonal_wave_ampl_phase(
    fc: xr.DataArray,
    phase_deg: bool = False,
    wave_coord: str = "",
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Calculates the amplitudes and relative phases of waves in the
    zonal direction.

    Parameters
    ----------
    fc : `xarray.DataArray`
        Fourier coefficients as a function of zonal wavenumber, as
        returned by `zonal_wave_coeffs`.
    phase_deg : boolean, optional
        Whether to return the relative phases in radians or degrees.
    wave_coord : str, optional
        The coordinate name of the wavenumber dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the wavenumbers

    Returns
    -------
    tuple of two `xarray.DataArray`
        Contains the amplitudes and phases for the input field zonal waves.

    See Also
    --------
    zonal_wave_coeffs

    """

    if wave_coord == "":
        coords = infer_xr_coord_names(fc, required=["zonal_wavenum"])
        wave_coord = coords["zonal_wavenum"]

    if "nlons" not in fc.attrs:
        msg = "input DataArray must have an 'nlons' attribute specifying the number of longitudes in the source data"
        raise KeyError(msg)

    # where the longitudinal wavenumber is 0, `where' will
    # mask to nan, so np.isfinite will return False in those
    # spots and true everywhere else. Thus, add 1 to get
    # the multiplying mask that keeps in mind the "zeroth"
    # mode (which would be the zonal mean, if kept)
    #
    # this is necessary because of the symmetric spectrum,
    # so all other wavenumbers except the 0th need to
    # be multipled by 2 to get the right amplitude
    mult_mask = np.isfinite(fc.where(fc[wave_coord] != 0)) + 1

    with xr.set_options(keep_attrs=True):
        ampl = np.absolute(fc) * mult_mask / fc.nlons
        # phas = np.angle(fc, deg=phase_deg) # returns a np.ndarray instead of xr.DataArray/Dataset
        phas = xr.apply_ufunc(np.angle, fc, kwargs={"deg": phase_deg})

    return (ampl.astype(fc.dtype), phas.astype(fc.dtype))


def filter_by_zonal_wave_truncation(
    fc: xr.DataArray,
    waves: Iterable[int],
    fftpkg: str = "scipy",
    wave_coord: str = "",
    lons: None | xr.DataArray = None,
) -> xr.DataArray:
    """Filters a field by truncating the zonal wavenumbers. This is done
    by taking an inverse rFFT of the Fourier coefficients, with the unwanted
    wavenumbers set to zero.

    Parameters
    ----------
    fc : `xarray.DataArray`
        Fourier coefficients as a function of zonal wavenumber, as
        returned by `zonal_wave_coeffs`.
    waves : iterable of int or slice
        The wavenumbers to keep. If a slice is given, it must be a slice
        of integers. If an iterable is given, it must be an iterable of
        integers.
    fftpkg : {'scipy', 'numpy'}, optional
        The FFT package to use. Defaults to 'scipy'.
    wave_coord : str, optional
        The coordinate name of the wavenumber dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the wavenumbers
    lons: `xarray.DataArray`, optional
        The longitude coordinate of the input field. If not given, the
        function will attempt to reconstruct the coordinate from the input
        Fourier coefficients, by looking for 'nlons' and 'lon0' attributes.

    """
    if wave_coord == "":
        coords = infer_xr_coord_names(fc, required=["zonal_wavenum"])
        wave_coord = coords["zonal_wavenum"]

    if "nlons" not in fc.attrs:
        msg = "input DataArray must have an 'nlons' attribute specifying the number of longitudes in the source data"
        raise KeyError(msg)

    if "lon0" not in fc.attrs:
        msg = "input DataArray must have a 'lon0' attribute specifying the starting longitude of the source data"
        raise KeyError(msg)

    wavenum_ax = fc.get_axis_num(wave_coord)
    if lons is None:
        warnings.warn("attempting to infer the input longitude coordinate", UserWarning)
        lons = lon_coord(
            360 / fc.attrs["nlons"],
            left_lim=fc.attrs["lon0"],
            right_lim=fc.attrs["lon0"] + 360.0,
        )

    if fftpkg == "scipy":
        new_dims = list(fc.dims)
        new_dims[wavenum_ax] = lons.name  # type: ignore
        new_coords = dict(fc.coords)
        new_coords.pop(wave_coord, None)
        new_coords[lons.name] = lons

        filtered = scipy.fft.irfft(
            fc.where(fc[wave_coord].isin(waves), other=0j).values, axis=wavenum_ax
        )
        filtered = xr.DataArray(filtered, dims=new_dims, coords=new_coords)

    elif fftpkg == "xrft":
        filtered = xrft.ifft(
            fc.where(fc[wave_coord].isin(waves), other=0j),
            dim=wave_coord,
            real_dim=wave_coord,
            true_amplitude=False,
            true_phase=False,
            prefix="",
        )
        filtered = filtered.rename({wave_coord: lons.name}).assign_coords(
            {lons.name: lons}
        )

    else:
        msg = "Invalid fftpkg: must be 'scipy' or 'xrft'"
        raise ValueError(msg)

    return filtered


def zonal_wave_contributions(
    fc: xr.DataArray,
    waves: Iterable[int],
    fftpkg: str = "scipy",
    wave_coord: str = "",
    lons: None | xr.DataArray = None,
) -> xr.DataArray:
    r"""Computes the individual contributions of each zonal wavenumber
    to the input field. This is done by taking an inverse FFT of the
    Fourier coefficients, with all but one wavenumber set to zero.

    Parameters
    ----------
    fc : `xarray.DataArray`
        Fourier coefficients as a function of zonal wavenumber, as
        returned by `zonal_wave_coeffs`.
    waves : iterable of int, optional
        The zonal wavenumbers to maintain in the output.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options
        are scipy or xrft. Specifying scipy uses some operations that are
        memory-eager and leverages scipy.fft.rfft. Specifying xrft should
        leverage the benefits of xarray/dask for large datasets by using
        xrft.fft. Defaults to scipy.
    wave_coord : str, optional
        The coordinate name of the wavenumber dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the wavenumbers
    lons: `xarray.DataArray`, optional
        The longitude coordinate of the input field. If not given, the
        function will attempt to reconstruct the coordinate from the input
        Fourier coefficients, by looking for 'nlons' and 'lon0' attributes.

    Returns
    -------
    `xarray.DataArray`
        The individual contributions of each wavenumber to the original
        input field.

    See Also
    --------
    zonal_wave_coeffs
    filter_by_zonal_wave_truncation

    """

    if wave_coord == "":
        coords = infer_xr_coord_names(fc, required=["zonal_wavenum"])
        wave_coord = coords["zonal_wavenum"]

    contributions = []
    for wave in waves:
        contributions.append(
            filter_by_zonal_wave_truncation(
                fc, [wave], fftpkg=fftpkg, wave_coord=wave_coord, lons=lons
            )
        )
    contributions = xr.concat(contributions, dim=wave_coord).assign_coords(
        {wave_coord: waves}
    )

    return contributions


def zonal_wave_covariance(
    fc1: xr.DataArray,
    fc2: xr.DataArray,
    wave_coord: str = "",
    nlons: int | None = None,
) -> xr.DataArray:
    r"""Calculates the covariance of two fields partititioned into
    zonal wavenumbers.

    Parameters
    ----------
    fc1 : `xarray.DataArray`
        The zonal Fourier coefficients of the first field, as returned by
        `zonal_wave_coeffs`.
    fc2 : `xarray.DataArray`
        The zonal Fourier coefficients of the second field, as returned by
        `zonal_wave_coeffs`.
    wave_coord : str optional
        The coordinate name of the wavenumber dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the wavenumbers.
    nlons: int, optional
        The number of longitudes in the original data. If not provided, the
        function will try to find this in the attrs of fc1 or fc2.

    Returns
    -------
    `xarray.DataArray`

    See Also
    --------
    zonal_wave_coeffs

    """

    # Ensure that input DataArrays are fully consistent
    xr.align(fc1, fc2, join="exact", copy=False)

    # If both DataArrays are fully consistent, then this block will cover both
    if wave_coord == "":
        coords = infer_xr_coord_names(fc1, required=["zonal_wavenum"])
        wave_coord = coords["zonal_wavenum"]

    if (nlons is None) and ("nlons" not in fc1.attrs) and ("nlons" not in fc2.attrs):
        raise ValueError(
            "nlons must either be provided as a kwarg or be in the attrs of fc1 or fc2"
        )
    elif nlons is None:
        nlons = fc1.attrs.get("nlons", fc2.attrs.get("nlons"))

    mult_mask = np.isfinite(fc1.where(fc1.zonal_wavenum != 0)) + 1
    cov = mult_mask * np.real(fc1 * fc2.conj()) / (nlons * nlons)  # type: ignore

    return cov
