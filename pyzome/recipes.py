import xarray as xr

from .basic import zonal_mean, zonal_wave_coeffs, zonal_wave_covariance
from .checks import infer_xr_coord_names


def _print_if_true(msg, condition, **kwargs):
    r"""Simple utility function to print only if the given condition is True.

    Parameters
    ----------
    msg : string
        The message to print
    condition : bool
        The boolean that determines whether
        anything is actually printed.

    """

    if (condition is True):
        print(msg, **kwargs)
    return


def create_zonal_mean_dataset(ds, verbose=False, include_waves=False,
                              waves=None, fftpkg="scipy", lon_coord=""):
    r"""Compiles a "zonal mean dataset".

    Given an xarray dataset containing full fields of basic state
    variables such as velocity components and temperatures, this
    function will compute as many zonal mean diagnostics as possible.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset containing full fields (i.e., containing latitude &
        longitude dimensions) of basic state variables. This function
        currently assumes specific names and units:

        'u' = zonal wind component in m/s
        'v' = meridional wind component in m/s
        'w' = vertical pressure velocity in Pa/s
        'T' = temperature in K
        'Z' = geopotential height in m

        If your data names, dimensions, and/or units do not conform to these
        restrictions, please change beforehand. Dimensions and names can
        easily be changed with the `rename` method of xarray Datasets/DataArrays.

        Note that ds need not contain all of these variables, and this
        function will still provide as many diagnostics as possible.

    verbose : bool, optional
        Whether to print out progress information as the function proceeds.
        Defaults to False.
    include_waves : bool, optional
        Whether to include possible longitudinal wave diagnostics such as
        eddy covariances and fourier coefficients. Defaults to False.
    waves : array-like, optional
        The specific zonal wavenumbers to maintain in the output. This
        kwarg is only considered if include_waves is True.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        'scipy' or 'xrft'. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.
        This kwarg is only considered if include_waves is True.

    Returns
    -------
    `xarray.Dataset`
        An xarray Dataset containing the possible zonal mean diagnostics.

    Notes
    -----
    Please see https://essd.copernicus.org/articles/10/1925/2018/ for
    a description of a different zonal mean dataset compiled for the
    SPARC Reanalysis Intercomparison Project. This function does *not*
    provide all the same diagnostics as listed in that publication.
    However, if this function is provided with all of u, v, w, and T,
    it will return all terms necessary from which further diagnostics
    can be computed to, for instance, perform zonal Eulerian and
    Transformed Eulerian Mean momentum budgets.

    """
    
    if lon_coord == "":
        coords = infer_xr_coord_names(ds, required=["lon"])
        lon_coord = coords["lon"]

    all_vars = ['u', 'v', 'w', 'T', 'Z']
    cov_pairs = [('u', 'v'), ('v', 'T'), ('u', 'w'), ('w', 'T')]
    wave_coeffs = ['T', 'Z']

    long_names = {
        'u': 'Zonal Mean Zonal Wind',
        'v': 'Zonal Mean Meridional Wind',
        'w': 'Zonal Mean Vertical Pressure Velocity',
        'T': 'Zonal Mean Temperature',
        'Z': 'Zonal Mean Geopotential Height',
        'uv': 'Total Eddy Momentum Flux',
        'vT': 'Total Eddy Heat Flux',
        'uw': 'Total Eddy Vertical Momentum Flux',
        'wT': 'Total Eddy Vertical Heat Flux',
        'uv_k': 'Eddy Momentum Flux due to Zonal Wave-k',
        'vT_k': 'Eddy Heat Flux due to Zonal Wave-k',
        'uw_k': 'Eddy Vertical Momentum Flux due to Zonal Wave-k',
        'wT_k': 'Eddy Vertical Heat Flux due to Zonal Wave-k',
        'Z_k_real': 'Real part of Fourier coefficients of Zonal Geohgt Waves',
        'Z_k_imag': 'Imaginary part of Fourier coefficients of Zonal Geohgt Waves',
        'T_k_real': 'Real part of Fourier coefficients of Zonal Temperature Waves',
        'T_k_imag': 'Imaginary part of Fourier coefficients of Zonal Temperature Waves'
    }

    units = {
        'u': 'm s-1',
        'v': 'm s-1',
        'w': 'Pa s-1',
        'T': 'K',
        'Z': 'm',
        'uv':  'm+2 s-2',
        'vT': 'K m s-1',
        'uw': 'm Pa s-2',
        'wT': 'K Pa s-1',
        'uv_k': 'm+2 s-2',
        'vT_k': 'K m s-1',
        'uw_k': 'm Pa s-2',
        'wT_k': 'K Pa s-1',
        'Z_k_real': 'm',
        'Z_k_imag': 'm',
        'T_k_real': 'K',
        'T_k_imag': 'K'
    }

    inter = {}

    _print_if_true('*** Compiling zonal means and eddies', verbose)
    for var in all_vars:
        if (var in ds.variables):
            _print_if_true(f'    {var}', verbose)

            zm = zonal_mean(ds[var])
            ed = ds[var] - zm

            inter[f'{var}'] = zm
            inter[f'{var}ed'] = ed
            out_coords = inter[f'{var}'].coords

    _print_if_true('*** Compiling zonal covariances', verbose)
    for var1, var2 in cov_pairs:
        if (var1 in ds.variables) and (var2 in ds.variables):
            _print_if_true(f'    {var1}{var2}', verbose)

            cov = zonal_mean(inter[f'{var1}ed'] * inter[f'{var2}ed'])
            inter[f'{var1}{var2}'] = cov

    if include_waves is True:

        _print_if_true('*** Compiling zonal wave covariances', verbose)
        for var1, var2 in cov_pairs:
            if (var1 in ds.variables) and (var2 in ds.variables):
                _print_if_true(f'    {var1}{var2}', verbose)

                cov = zonal_wave_covariance(ds[var1], ds[var2], waves=waves, fftpkg=fftpkg)
                inter[f'{var1}{var2}_k'] = cov
                out_coords = inter[f'{var1}{var2}_k'].coords

        _print_if_true('*** Compiling zonal wave Fourier coefficients', verbose)
        for var in wave_coeffs:
            if (var in ds.variables):
                _print_if_true(f'    {var}', verbose)

                fc = zonal_wave_coeffs(ds[var], waves=waves, fftpkg=fftpkg)
                inter[f'{var}_k_real'] = fc.real
                inter[f'{var}_k_imag'] = fc.imag
                out_coords = inter[f'{var}_k_real'].coords

    # Remove the eddy fields
    out_vars = list(inter.keys())
    for var in out_vars:
        if 'ed' in var:
            inter.pop(var)

    # Ascribe names and long_name attributes to each DataArray
    # and create the encoding dictionary to use
    out_vars = inter.keys()
    for var in out_vars:
        inter[var].name = var
        inter[var].attrs['long_name'] = long_names[var]
        inter[var].attrs['units'] = units[var]

    out_ds = xr.Dataset(inter, coords=out_coords)
    out_ds.attrs['nlons'] = ds[lon_coord].size

    return out_ds
