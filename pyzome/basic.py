import numpy as np
import xarray as xr
import scipy
import xrft


def zonal_mean(dat):
    r"""Compute the zonal mean.

    This is primarily a convenience function that will make other
    code more explicit/readable.

    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a dimension named longitude that spans all 360 degrees

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        the mean across the longitude dimension

    TO DO
    -----
    * Do not assume the 'longitude' dimension name

    """
    return dat.mean('longitude')


def meridional_mean(dat, lat1, lat2):
    r"""Compute the cos(lat) weighted mean of a quantity between two latitudes.

    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a dimension named latitude that spans
        lat1 and lat2

    lat1 : float
        The beginning latitude limit of the band average

    lat2 : float
        The ending latitude limit of the band average

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        the weighted mean across the latitude dimension limited
        by lat1 and lat2

    Notes
    -----
    At present this function uses a slice for limiting the
    latitudes, and does not check for ordering. This means
    to get a proper result, you must know the ordering
    of the latitudes of your data. If they are oriented
    North to South (such as going from 90N to 90S), then
    lat1 should be greater than lat2. If they are oriented
    South to North (such as going from 90S to 90N), then
    lat1 should be less than lat2.

    TO DO
    -----
    * Do not assume the 'latitude' dimension name
    * Check latitude ordering and throw an error
      if the latitudes in the data do not contain
      all/part of the latitude range.

    """

    wgts = np.cos(np.deg2rad(dat.latitude.sel(latitude=slice(lat1,lat2))))

    return dat.sel(latitude=slice(lat1, lat2)).weighted(wgts).mean('latitude')


def zonal_wave_coeffs(dat, *, waves=None, fftpkg='scipy'):
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    This is a primarily a driver function that shifts the data depending
    on the specified fftpkg.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension, for specified waves only.

    TO DO
    -----
    * Do not assume the 'longitude' dimension name

    """

    if fftpkg not in ['scipy', 'xrft']:
        msg = 'fftpkg keyword arg must be one of scipy or xarray'
        raise ValueError(msg)

    funcs = {
        'scipy': _zonal_wave_coeffs_scipy,
        'xrft': _zonal_wave_coeffs_xrft
    }

    nlons = dat.longitude.size

    fc = funcs[fftpkg](dat)

    fc.attrs['nlons'] = nlons
    fc.attrs['lon0'] = dat.longitude.values[0]
    if (waves is not None):
        fc = fc.sel(lon_wavenum=waves)

    return fc


def _zonal_wave_coeffs_scipy(dat):
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    Uses scipy.fft.rfft to perform the calculation.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.

    TO DO
    -----
    * Do not assume the 'longitude' dimension name

    """
    nlons = dat.longitude.size
    lon_ax = dat.get_axis_num('longitude')

    new_dims = list(dat.dims)
    new_dims[lon_ax] = 'lon_wavenum'

    new_coords = dict(dat.coords)
    new_coords.pop('longitude')
    new_coords['lon_wavenum'] = np.arange(0, nlons//2 + 1)

    fc = scipy.fft.rfft(dat.values, axis=lon_ax)
    fc = xr.DataArray(fc, coords=new_coords, dims=new_dims)

    return fc


def _zonal_wave_coeffs_xrft(dat):
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    Uses xrft.fft to perform the calculation.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.

    TO DO
    -----
    * Do not assume the 'longitude' dimension name

    """

    fc = xrft.fft(dat, dim='longitude', real_dim='longitude',
                  true_phase=False, true_amplitude=False)

    fc = fc.rename({'freq_longitude': 'lon_wavenum'})
    fc = fc.assign_coords({'lon_wavenum': np.arange(fc.lon_wavenum.size)})

    return fc


def zonal_wave_ampl_phase(dat, waves=None, phase_deg=False, fftpkg='scipy'):
    r"""Calculates the amplitudes and relative phases of waves in the zonal direction.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    phase_deg : boolean, optional
        Whether to return the relative phases in radians or degrees.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (amplitudes, phases)

    See Also
    --------
    zonal_wave_coeffs

    """

    fc = zonal_wave_coeffs(dat, waves=waves, fftpkg=fftpkg)

    # where the longitudinal wavenumber is 0, `where' will
    # mask to nan, so np.isfinite will return False in those
    # spots and true everywhere else. Thus, add 1 to get
    # the multiplying mask that keeps in mind the "zeroth"
    # mode (which would be the zonal mean, if kept)
    #
    # this is necessary because of the symmetric spectrum,
    # so all other wavenumbers except the 0th need to
    # be multipled by 2 to get the right amplitude
    mult_mask = np.isfinite(fc.where(fc.lon_wavenum != 0)) + 1

    ampl = mult_mask*np.abs(fc) / fc.nlons
    phas = np.angle(fc, deg=phase_deg)

    return (ampl.astype(dat.dtype), phas.astype(dat.dtype))


def zonal_wave_contributions(dat, waves=None, fftpkg='scipy'):
    r"""Computes contributions of waves with zonal wavenumber k to the input field.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.

    Returns
    -------
    `xarray.DataArray`

    See Also
    --------
    zonal_wave_coeffs

    TO DO
    -----
    * Do not assume the 'longitude' dimension name

    """
    fc = zonal_wave_coeffs(dat, waves=waves, fftpkg=fftpkg)

    if (waves is None):
        waves = fc.lon_wavenum.values

    recons = []
    if (fftpkg == 'scipy'):
        new_dims = list(dat.dims)
        new_dims += ['lon_wavenum']
        new_coords = dict(dat.coords)
        new_coords['lon_wavenum'] = waves

        for k in waves:
            mask = np.isnan(fc.where(fc.lon_wavenum != k))

            kcont = scipy.fft.irfft((fc*mask).values, axis=fc.get_axis_num('lon_wavenum'))
            recons.append(kcont[..., np.newaxis])

        recons = np.concatenate(recons, axis=-1)
        recons = xr.DataArray(recons, dims=new_dims, coords=new_coords)

    elif (fftpkg == 'xarray'):
        fc = fc.rename({'lon_wavenum': 'freq_longitude'})

        for k in waves:
            mask = np.isnan(fc.where(fc.lon_wavenum != k))

            kcont = xrft.ifft((fc*mask).values, dim='lon_wavenum', real_dim='lon_wavenum')
            recons.append(kcont)

        recons = xr.concat(recons, dim='lon_wavenum')
        recons = recons.assign_coords({'lon_wavenum': waves, 'longitude': dat.longitude})

    return recons.astype(dat.dtype)


def zonal_wave_covariance(dat1, dat2, waves=None, fftpkg='scipy'):
    r"""Calculates the covariance of two fields partititioned into zonal wavenumbers.

    Parameters
    ----------
    dat1 : `xarray.DataArray`
        field containing a dimension named longitude that spans all 360 degrees.
        Should have the same shape as dat2.
    dat2 : `xarray.DataArray`
        another field also containing a dimension named longitude that spans all
        360 degrees. Should have the same shape as dat1.
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.

    Returns
    -------
    `xarray.DataArray`

    See Also
    --------
    zonal_wave_coeffs

    TO DO
    -----
    * Do not assume the 'longitude' dimension name
    * Check for consistency between dat1 and dat2 and throw errors

    """

    nlons = dat1['longitude'].size

    fc1 = zonal_wave_coeffs(dat1, waves=waves, fftpkg=fftpkg)
    fc2 = zonal_wave_coeffs(dat2, waves=waves, fftpkg=fftpkg)

    mult_mask = np.isfinite(fc1.where(fc1.lon_wavenum != 0)) + 1
    cov = mult_mask*np.real(fc1 * fc2.conj())/(nlons**2)

    return cov
