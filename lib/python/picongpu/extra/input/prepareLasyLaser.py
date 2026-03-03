"""
This file is part of PIConGPU.

Copyright 2025-2026 Edgar Marquardt

Get the full electric field from a Lasy laser and save it to an openPMD compatible file in a way that PIConGPU understands.

Some of the code taken from lasy 0.6.2 and then modified. Those parts are marked."""

from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import os
import openpmd_api as io

from lasy import __version__ as lasy_version
from lasy.utils.grid import Grid
from lasy.profiles.profile import Profile
from lasy.laser import Laser

try:
    from tqdm.auto import tqdm

    tqdm_available = True
    bar_format = "{l_bar}{bar}| {elapsed}<{remaining} [{rate_fmt}{postfix}]"
except Exception:
    tqdm_available = False


def get_full_field(laser, theta=0, Nt=None, Nr=None, Nx=None, Ny=None, forced_dt=None, offset_frac=0):
    """
    Reconstruct the laser pulse with carrier frequency on the default grid.

    Parameters
    ----------
    laser : laser laser object
        The laser for which the full field needs to be calculated.

    Nt : int (optional)
        Number of time points on which field should be sampled. If is None,
        the orignal time grid is used, otherwise field is interpolated on a
        new grid.

    Nx : int (optional)
        Number of x-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    Ny : int (optional)
        Number of y-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    Nr : int (optional)
        Number of r-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    theta : float (optional)
        for the dim=rt case. defines the angle around the pulse the field
        should be calculated in.

    froced_dt : float (optional)
        if not None it forces dt to be this value, if possible.
        if forced_dt and Nt is given the section is taken from the end of the field.

    offset_frac : float between 0 and 1 (optional)
        only relevant, when both Nt and forced_dt are given. offsets the Nt * forced_dt
        range from the upper end. This is interpreted as the fraction of the original field
        the offset is supposed to be.

    Returns
    -------
        Et : ndarray (V/m)
            The reconstructed field, with shape (Nr, Nt) (for `rt`)
            or (Nx, Ny, Nt) (for `xyt`).
        extent : ndarray (Tmin, Tmax, Xmin, Xmax, [Ymin, Ymax])
            Physical extent of the reconstructed field.
    """
    omega0 = laser.profile.omega0
    env = laser.grid.get_temporal_field()
    time_axis = laser.grid.axes[-1]

    # If the field is not an envelope, it is a full field, so no
    # reason to recompute the full field.
    assert laser.grid.is_envelope
    if laser.dim == "rt":
        # Cut off everything beyond Nr
        if Nr is None:
            Nr = env.shape[1]
            frac = 1.0
        else:
            frac = env.shape[1] / Nr
        env = env[:, :Nr, :]

        # cut the laser at the right angle and prepare the field accordingly
        # taken from Lasy
        azimuthal_phase = np.exp(-1j * laser.grid.azimuthal_modes * theta)
        env_upper = env * azimuthal_phase[:, None, None]
        env_upper = env_upper.sum(0)
        azimuthal_phase = np.exp(1j * laser.grid.azimuthal_modes * theta)
        env_lower = env * azimuthal_phase[:, None, None]
        env_lower = env_lower.sum(0)
        env = np.vstack((env_lower[::-1][:-1], env_upper))

        if Nt is not None or forced_dt is not None:
            # Now interpolation is required.
            # First preparation of the new time axis
            if Nt is not None and forced_dt is not None:
                if offset_frac > 0:
                    assert forced_dt * Nt < (laser.grid.hi[-1] - laser.grid.lo[-1]) * (1 - offset_frac), (
                        "forced_dt * Nt does not fit into field"
                    )
                    p = forced_dt * Nt / (laser.grid.hi[-1] - laser.grid.lo[-1])
                    print(
                        "Warning: Nt and forced_dt given. Can not return the entire field, just "
                        + str(100 * p)[:5]
                        + " percent."
                    )
                    print("Offsetting by " + str(100 * offset_frac)[:5] + " percent of the original field")
                    time_axis_new = (
                        np.arange(Nt) * forced_dt
                        + laser.grid.hi[-1]
                        - Nt * forced_dt
                        - offset_frac * (laser.grid.hi[-1] - laser.grid.lo[-1])
                    )
                else:
                    assert forced_dt * Nt < (laser.grid.hi[-1] - laser.grid.lo[-1]), (
                        "forced_dt * Nt does not fit into field"
                    )
                    p = forced_dt * Nt / (laser.grid.hi[-1] - laser.grid.lo[-1])
                    print(
                        "Warning: Nt and forced_dt given. Can not return the entire field, just "
                        + str(100 * p)[:5]
                        + " percent."
                    )
                    time_axis_new = np.arange(Nt) * forced_dt + laser.grid.hi[-1] - Nt * forced_dt
            elif forced_dt is None:
                time_axis_new, dt = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], Nt, retstep=True)
                print("dt =", dt)
            elif Nt is None:
                time_axis_new = (
                    np.arange(int((laser.grid.hi[-1] - laser.grid.lo[-1]) / forced_dt)) * forced_dt + laser.grid.lo[-1]
                )
                Nt = len(time_axis_new)
                print("Nt =", Nt)
            # prepare new field
            env_new = np.zeros((2 * Nr - 1, Nt), dtype=env.dtype)
            if tqdm_available:
                pbar = tqdm(total=2 * Nr - 1, bar_format=bar_format)
            # and then interpolate the envelope for Nt and forced_dt
            # taken from Lasy
            for ir in range(2 * Nr - 1):
                interp_fu_abs = interp1d(time_axis, np.abs(env[ir]))
                slice_abs = interp_fu_abs(time_axis_new)
                interp_fu_angl = interp1d(time_axis, np.unwrap(np.angle(env[ir])))
                slice_angl = interp_fu_angl(time_axis_new)
                env_new[ir] = slice_abs * np.exp(1j * slice_angl)
                if tqdm_available:
                    pbar.update(1)
                else:
                    if ir % 20 == 19:
                        print(ir + 1, "out of", 2 * Nr - 1)

            if tqdm_available:
                pbar.close()

            # act as if nothing happend so the rest of the function does its thing
            time_axis = time_axis_new
            env = env_new

    else:
        # Cut out the section of size Nx, Ny, Nz
        if Nx is not None:
            midx = env.shape[0] // 2
            xmin = midx - Nx // 2
            xmax = midx + Nx // 2
            fracx = env.shape[0] / Nx
            env = env[xmin:xmax, :, :]
            midx = (laser.grid.hi[0] + laser.grid.lo[0]) / 2
            xlo = midx - (laser.grid.hi[0] - laser.grid.lo[0]) / 2 / fracx
            xhi = midx + (laser.grid.hi[0] - laser.grid.lo[0]) / 2 / fracx
        else:
            Nx = env.shape[0]
            fracx = 1.0
            xlo = laser.grid.lo[0]
            xhi = laser.grid.hi[0]
        if Ny is not None:
            midy = env.shape[1] // 2
            ymin = midy - Ny // 2
            ymax = midy + Ny // 2
            fracy = env.shape[1] / Ny
            env = env[:, ymin:ymax, :]
            midy = (laser.grid.hi[1] + laser.grid.lo[1]) / 2
            ylo = midy - (laser.grid.hi[1] - laser.grid.lo[1]) / 2 / fracy
            yhi = midy + (laser.grid.hi[1] - laser.grid.lo[1]) / 2 / fracy
        else:
            Ny = env.shape[1]
            fracy = 1.0
            ylo = laser.grid.lo[1]
            yhi = laser.grid.hi[1]

        if Nt is not None or forced_dt is not None:
            # Now interpolation is required.
            # First preparation of the new time axis
            if Nt is not None and forced_dt is not None:
                if offset_frac > 0:
                    assert forced_dt * Nt < (laser.grid.hi[-1] - laser.grid.lo[-1]) * (1 - offset_frac), (
                        "forced_dt * Nt does not fit into field"
                    )
                    p = forced_dt * Nt / (laser.grid.hi[-1] - laser.grid.lo[-1])
                    print(
                        "Warning: Nt and forced_dt given. Can not return the entire field, just "
                        + str(100 * p)[:5]
                        + " percent."
                    )
                    print("Offsetting by " + str(offset_frac * 100)[:5] + " percent of the original field")
                    time_axis_new = (
                        np.arange(Nt) * forced_dt
                        + laser.grid.hi[-1]
                        - Nt * forced_dt
                        - offset_frac * (laser.grid.hi[-1] - laser.grid.lo[-1])
                    )
                else:
                    assert forced_dt * Nt < (laser.grid.hi[-1] - laser.grid.lo[-1]), (
                        "forced_dt * Nt does not fit into field"
                    )
                    p = forced_dt * Nt / (laser.grid.hi[-1] - laser.grid.lo[-1])
                    print(
                        "Warning: Nt and forced_dt given. Can not return the entire field, just "
                        + str(100 * p)[:5]
                        + " percent."
                    )
                    time_axis_new = np.arange(Nt) * forced_dt + laser.grid.hi[-1] - Nt * forced_dt
            elif forced_dt is None:
                time_axis_new, dt = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], Nt, retstep=True)
                print("dt =", dt)
            elif Nt is None:
                time_axis_new = (
                    np.arange(int((laser.grid.hi[-1] - laser.grid.lo[-1]) / forced_dt)) * forced_dt + laser.grid.lo[-1]
                )
                Nt = len(time_axis_new)
                print("Nt =", Nt)
            # prepare new field
            env_new = np.zeros((Nx, Ny, Nt), dtype=env.dtype)
            if tqdm_available:
                pbar = tqdm(total=Nx, bar_format=bar_format)
            # and then interpolate the envelope for Nt and forced_dt
            # taken from Lasy
            for ix in range(Nx):
                for iy in range(Ny):
                    interp_fu_abs = interp1d(time_axis, np.abs(env[ix, iy]))
                    slice_abs = interp_fu_abs(time_axis_new)
                    interp_fu_angl = interp1d(time_axis, np.unwrap(np.angle(env[ix, iy])))
                    slice_angl = interp_fu_angl(time_axis_new)
                    env_new[ix, iy] = slice_abs * np.exp(1j * slice_angl)
                if tqdm_available:
                    pbar.update(1)
                else:
                    if ix % 20 == 19:
                        print(ix + 1, "out of", Nx)
            if tqdm_available:
                pbar.close()

            # act as if nothing happend so the rest of the function does its thing
            time_axis = time_axis_new
            env = env_new

    # applying the base frequency
    # taken from Lasy
    env *= np.exp(-1j * omega0 * time_axis[None, :])
    env = np.real(env)

    # getting the extent of the electric field

    if laser.dim == "rt":
        ext = np.array([np.min(time_axis), np.max(time_axis), -laser.grid.hi[0] / frac, laser.grid.hi[0] / frac])
    else:
        ext = np.array([np.min(time_axis), np.max(time_axis), ylo, yhi, xlo, xhi])

    return env, ext


def _gauss(x, mu, sigma, A):
    """fit function for gauss fitting"""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def get_tpeak(laser, pos_offset=False, method="stat", nx=None, ny=None, nr=None):
    """calculates the position of the intensity peak

    Parameters:
    laser : lasy laser object
        The laser for which to calculate the peak time.

    pos_offset : bool (optional)
        Wether to take the grid position into account.

    method : str in {"stat", "fit", "max", "avmax"} (optiona)
        The method of calculation.
        "stat": calculates the statistical average of the intensity on the axis.
        "avstat": calculates the statistical average of the intensity.
        "fit": fits a gaussian onto the intensity on axis.
        "avfit": fits a gaussian onto the intensity.
        "max": finds the maximum value in the laser field and returns its time.
        "avmax": like "max" but averages the transversal directionas first.

    nx, ny, nr : int (optional)
        if the method does not start with "av" this specifies the xy coordinate or the r respectively,
        where the t_peak is calculated. These are the indices for the coordiates.

    Returns:
    t_peak : float
        the intensity peak time.
    """
    assert method in ["stat", "avstat", "fit", "avfit", "max", "avmax"], "the given method does not exist."
    field = laser.grid.get_temporal_field()
    l_int = np.abs(field) ** 2
    s_int = np.sum(l_int, axis=(0, 1))
    if laser.dim == "xyt":
        assert (ny is None) == (nx is None), "not only one of nx and ny can be specified."
        if nx is not None:
            lineout = l_int[nx, ny, :]
        else:
            Nx, Ny = l_int.shape[0] // 2, l_int.shape[1] // 2
            lineout = l_int[Nx, Ny, :]
    if laser.dim == "rt":
        if nr is not None:
            lineout = l_int[0, nr, :]
        else:
            lineout = l_int[0, 0, :]
    match method:
        case "stat":
            t_peak = np.average(laser.grid.axes[-1], weights=lineout)
        case "avstat":
            t_peak = np.average(laser.grid.axes[-1], weights=s_int)
        case "fit":
            p0 = (0, (laser.grid.axes[-1][-1] - laser.grid.axes[-1][0]) / 4, np.max(lineout))
            coeff, _ = curve_fit(_gauss, laser.grid.axes[-1], lineout, p0=p0)
            t_peak = coeff[0]
        case "avfit":
            p0 = (0, (laser.grid.axes[-1][-1] - laser.grid.axes[-1][0]) / 4, np.max(s_int))
            coeff, _ = curve_fit(_gauss, laser.grid.axes[-1], s_int, p0=p0)
            t_peak = coeff[0]
        case "max":
            t_peak = laser.grid.axes[-1][np.argmax(lineout)]
        case "avmax":
            t_peak = laser.grid.axes[-1][np.argmax(s_int)]
        case _:
            print("?")
            raise ValueError
    if pos_offset:
        t_peak += laser.grid.position / c
    return t_peak


def cfl_condition(xi=0.995, dx=0.1772e-6, dy=0.4430e-7, dz=0.1772e-6, dt=None):
    """returns and prints the ideal time step given the spacing to get close to the
    limits of the Courant-Friedrich-Levy condition.

    Parameters:
    xi : float
        How close to the condition do you want to be? 1 is at the condition, 0 is dt=0

    dx, dy, dz : float
        the spacing in the three directions.

    dt : float (optional)
        if not None this will print, how far off this dt is from the ideal value

    Returns:
    dt_cfl : (float)
        the ideal dt given the spacing
    """
    dt_cfl = xi / np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2) / c
    print("dt should be:", dt_cfl)
    if dt is not None:
        print("error:", (dt / dt_cfl - 1) * 100, "%")
    return dt_cfl


class _PartialGrid(Grid):
    """partial grid class to save on memory. Do not use outside these functions.
    It does not work as intended otherwise as it does not have a spectral field.

    Code directly taken from lasy."""

    def __init__(
        self,
        dim,
        lo,
        hi,
        npoints,
        n_azimuthal_modes=None,
        is_envelope=True,
        is_cw=False,
        is_plane_wave=False,
        position=0.0,
    ):
        # Metadata
        ndims = 2 if dim == "rt" else 3
        assert dim in ["rt", "xyt"]
        assert len(lo) == ndims
        assert len(hi) == ndims

        lo = list(lo)
        hi = list(hi)
        npoints = list(npoints)

        if is_cw:
            if npoints[-1] != 1:
                print("CW profile: overwrite npoints to only 1 cell in the longitudinal direction.")
            lo[-1] = -0.5
            hi[-1] = 0.5
            npoints[-1] = 1
        if is_plane_wave:
            if npoints[0] != 1:
                print("Plane wave: overwrite npoints to only 1 cell in the transverse directions.")
            if dim == "rt":
                lo[0] = 0.0
                hi[0] = np.sqrt(1 / np.pi)
                npoints[0] = 1
            else:
                lo[0] = -0.5
                hi[0] = 0.5
                npoints[0] = 1
                lo[1] = -0.5
                hi[1] = 0.5
                npoints[1] = 1

        self.npoints = npoints
        self.axes = []
        self.dx = []
        for i in range(ndims):
            self.axes.append(np.linspace(lo[i], hi[i], npoints[i]))
            if len(self.axes[i]) > 1:
                self.dx.append(self.axes[i][1] - self.axes[i][0])
            else:
                self.dx.append(hi[i] - lo[i])

        self.lo = lo
        self.hi = hi

        if dim == "rt":
            self.n_azimuthal_modes = n_azimuthal_modes
            self.azimuthal_modes = np.r_[np.arange(n_azimuthal_modes), np.arange(-n_azimuthal_modes + 1, 0, 1)]

        # Data
        if dim == "xyt":
            self.shape = self.npoints
        elif dim == "rt":
            # Azimuthal modes are arranged in the following order:
            # 0, 1, 2, ..., n_azimuthal_modes-1, -n_azimuthal_modes+1, ..., -1
            ncomp = 2 * self.n_azimuthal_modes - 1
            self.shape = (ncomp, self.npoints[0], self.npoints[1])

        self.set_is_envelope(is_envelope)
        self.temporal_field = np.zeros(self.shape, dtype=self.dtype)
        self.temporal_field_valid = False
        self.spectral_field_valid = False
        self.position = position


class _PartialLaser(Laser):
    """partial laser class to save on memory during creation. Does not evaluate the profile.
    Also uses _PartialGrid to save on memory by not having the spectral field.

    Code directly taken from Lasy"""

    def __init__(self, dim, lo, hi, npoints, profile, n_azimuthal_modes=1, n_theta_evals=None):
        self.grid = _PartialGrid(
            dim,
            lo,
            hi,
            npoints,
            n_azimuthal_modes,
            is_cw=profile.is_cw,
            is_plane_wave=profile.is_plane_wave,
        )
        self.dim = dim
        self.profile = profile


def _rt_to_xyt(laser, Nx, Ny, points_between_r=1):
    """turns the laser to xyt for saving in openPMD
    preserves the point distances, not the hi and lo"""
    field = laser.grid.get_temporal_field()
    assert field.shape[0] == 1, "only supports 1 azimuthal mode"
    assert np.sqrt((Nx / 2) ** 2 + (Ny / 2) ** 2) / points_between_r < field.shape[1] - 1, (
        "Nx and Ny don't fit into the laser field"
    )

    # calculate the new hi and lo
    xfrac = Nx / 2 / field.shape[1] / points_between_r
    yfrac = Ny / 2 / field.shape[1] / points_between_r

    lo = (-xfrac * laser.grid.hi[0], -yfrac * laser.grid.hi[0], laser.grid.lo[1])
    hi = (xfrac * laser.grid.hi[0], yfrac * laser.grid.hi[0], laser.grid.hi[1])

    # interpolate the field to get a 3D representation
    field_new = np.zeros((Nx, Ny, field.shape[-1]), dtype=field.dtype)
    if tqdm_available:
        pbar = tqdm(total=Nx, bar_format=bar_format)
    for ix in range(Nx):
        for iy in range(Ny):
            r = np.sqrt((ix - Nx / 2) ** 2 + (iy - Ny / 2) ** 2) / points_between_r
            frac = r - int(r)
            field_new[ix, iy, :] = frac * field[0, int(r), :] + (1 - frac) * field[0, int(r) + 1, :]
        if tqdm_available:
            pbar.update(1)

    # make a partial laser for the return, containing the new field
    laser_new = _PartialLaser("xyt", lo, hi, field_new.shape, Profile(laser.profile.lambda0, laser.profile.pol))
    laser_new.grid.set_temporal_field(field_new)
    if tqdm_available:
        pbar.close()
    return laser_new


def _cut_first_frac(laser, cut_frac):
    """cuts the first points in t direction"""
    assert cut_frac < 1, "Can't cut the entire field"
    field = laser.grid.get_temporal_field()

    # prep
    N = int(field.shape[-1] * cut_frac)
    if np.max(np.abs(field[:, :, :N])) > 0.01 * np.max(np.abs(field)):
        print("Warning: Only cut unneccessary stuff")
        print(np.max(np.abs(field[:, :, :N])) / np.max(np.abs(field)))

    # cut the field
    field_new = field[:, :, N:]
    lo = laser.grid.lo
    lo[-1] += (laser.grid.hi[-1] - laser.grid.lo[-1]) * cut_frac

    # make a partial laser for the return, containing the new field
    laser_new = _PartialLaser(
        laser.dim, lo, laser.grid.hi, field_new.shape, Profile(laser.profile.lambda0, laser.profile.pol)
    )
    laser_new.grid.set_temporal_field(field_new)
    return laser_new


def write_to_openpmd_file(
    write_dir,
    file_prefix,
    file_format,
    array,
    extent,
    wavelength,
    pol,
    iteration=0,
    data_step=1,
    append=False,
    ittime=None,
):
    """
    Write the laser field into an openPMD file.

    This code was taken from Lasy and then modified to fit the needs of PIConGPU.

    Parameters
    ----------
    file_prefix : string
        The file name will start with this prefix.

    write_dir : string
        The directory where the file will be written.

    file_format : string
        Format to be used for the output file. Options are "h5" and "bp".

    iteration : int (optional)
        The iteration number for the file to be written.

    array : numpy ndarray, 3-dimensional with xyt
        The 3-dimensional array containing the full electric field.

    extent : numpy array of float
        Low and high ends of the electric field portrayed in array. Should be in SI units.
        np.array([time_low, time_high, y_low, y_high, x_low, x_high])

    wavelength : scalar
        Central wavelength for which the laser pulse envelope is defined.

    pol : list of 2 complex numbers
        Polarization vector that multiplies array to get the Ex and Ey arrays.

    data_step : int (optional)
        Only saves every (data_step)th data point to the file transversally.

    append : bool (optional)
        needs to be set to True, if the file should be added to not overwritten.

    ittime : float (optional)
        sets the time of the saved iteration. Otherwise the number of iterations so far is used.
    """
    # Create file and open it as openPMD series
    full_filepath = os.path.join(write_dir, "{}.{}".format(file_prefix, file_format))
    os.makedirs(write_dir, exist_ok=True)
    if append:
        series = io.Series(full_filepath, io.Access.append)
    else:
        series = io.Series(full_filepath, io.Access.create)
    series.set_software("lasy", lasy_version)

    # create the iteration
    i = series.iterations[iteration]
    if ittime is not None:
        i.set_attribute("time", ittime)
    else:
        if append:
            i.set_attribute("time", len(series.iterations) - 1)
        else:
            i.set_attribute("time", 0)

    # translate t to z
    # Because PIConGPU's FromOpenPMDPulse translates it back this is ok.
    extent[0] *= c
    extent[1] *= c

    array = array[::data_step, ::data_step, :]

    # Define the mesh
    m = i.meshes["E"]
    m.grid_spacing = [
        (hi - lo) / (npoints - 1) for hi, lo, npoints in zip(extent[1::2], extent[0::2], array.shape[2::-1])
    ]

    m.grid_global_offset = [0, 0, 0]
    m.geometry = io.Geometry.cartesian
    m.axis_labels = ["z", "y", "x"]

    i.dt = m.grid_spacing[0] / c

    # Store metadata needed to reconstruct the field
    m.set_attribute("angularFrequency", 2 * np.pi * c / wavelength)
    m.unit_dimension = {
        io.Unit_Dimension.M: 1,
        io.Unit_Dimension.L: 1,
        io.Unit_Dimension.I: -1,
        io.Unit_Dimension.T: -3,
    }

    # Switch from x,y,t (internal to lasy) to t,y,x (in openPMD file)
    # This is because many PIC codes expect x to be the fastest index
    print("transposing")
    data = np.transpose(array).astype(np.float32)
    print("Data type:", data.dtype)

    # Define the dataset
    dataset = io.Dataset(data.dtype, data.shape)
    if pol[0] > pol[1]:
        E_pol = m["x"]
        E_trans = m["y"]
    else:
        E_pol = m["y"]
        E_trans = m["x"]
    E_z = m["z"]

    E_pol.reset_dataset(dataset)
    E_trans.reset_dataset(dataset)
    E_z.reset_dataset(dataset)

    # put in the data
    E_pol[:, :, :] = data
    E_trans.make_constant(0.0)
    E_z.make_constant(0.0)
    print("flushing")
    series.flush()
    series.close()


def show(
    array,
    extent,
    cutfrac=0.5,
    cutdim=2,
    transpose=False,
    linthresh_frac=0.01,
    xlabel="",
    ylabel="",
    alabel="",
    title="",
    flipx=False,
    figsize=(16, 12),
    ret_ax=False,
):
    """shows 2D or a slice through 3D data on a nice symlog plot.

    Parameters:
    array : numpy ndarray (2 or 3 dimensional)
        The array containing the data.

    extent : list of 2 or 3 lists of 2 float
        low end then high end of each dimension.

    cutdim : int {0,1,2} (optional)
        Only relevant when using a 3D array.
        defines the dimension of the array going out of the screen in the plot.

    cutfrac : float [0,1] (optional)
        Only relevant when using a 3D array.
        defines where along the cutdim the plot should be in terms of fraction along it.

    transpose : bool (optional)
        transpose the array before diplaying it.

    linthresh_frac : float (optional)
        fraction of the absolute maximum from where the normalistion should be linear.

    xlabel, ylabel : str (optional)
        labels for the plot axes.

    alabel : str (optional)
        label for the colorbar describing the arraydata.

    title : str (optional)
        title for the plot.

    flipx : bool (optional)
        flips the x direction of the plot

    figsize : tuple of float (optional)
        Sets the size of the matplotlib figure.

    ret_ax : bool (optional)
        Whether to return the axes oject and the fig object instead of showing it.

    Returns:
    fig : pyplot figure object (optional)
        only returned if ret_ax is True.

    ax : pyplot axes object (optional)
        only returned if ret_ax is True.
    """
    # 1. cut
    if len(array.shape) > 2:
        assert len(array.shape) == 3
        if cutdim == 0:
            array = array[int(array.shape[0] * cutfrac), :, :]
            extent = extent[1:]
        elif cutdim == 1:
            array = array[:, int(array.shape[1] * cutfrac), :]
            extent = [extent[0], extent[2]]
        elif cutdim == 2:
            array = array[:, :, int(array.shape[2] * cutfrac)]
            extent = extent[:2]
        else:
            raise ValueError("cutdim must be in {0,1,2}")

    # 2. find the limits and make the axes
    m = np.max(np.abs(array))
    if m <= 0:
        print("nothing to see here")
        return

    x = np.linspace(extent[0][0], extent[0][1], array.shape[0] + 1, endpoint=True)
    y = np.linspace(extent[1][0], extent[1][1], array.shape[1] + 1, endpoint=True)

    # 3. transpose
    if transpose:
        array = array.transpose()
        x, y = y, x

    if flipx:
        array = np.flip(array, 1)

    # 3. plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    pcm = ax.pcolormesh(y, x, array, norm=clr.SymLogNorm(linthresh_frac * m, vmin=-m, vmax=m), cmap="seismic")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb = fig.colorbar(pcm, ax=ax, extend="both")
    cb.set_label(alabel)

    if ret_ax:
        return fig, ax

    fig.show()
    plt.show()


def _show(f, ext, show_SI, linthresh_frac, title, ret_ax):
    """internal function to show the field"""
    if show_SI:
        xlabel = "t/s"
        ylabel = "x/m"
        if len(ext) == 4:
            extent = [[ext[2], ext[3]], [ext[0], ext[1]]]
        elif len(ext) == 6:
            extent = [[ext[4], ext[5]], [ext[2], ext[3]], [ext[0], ext[1]]]
        else:
            raise ValueError("ext must be list of length 4 for rt or 6 for xyt")
    else:
        xlabel = "t"
        ylabel = "x"
        extent = [[0, sh] for sh in f.shape[::-1]]

    return show(
        f,
        extent,
        cutdim=1,
        linthresh_frac=linthresh_frac,
        xlabel=xlabel,
        ylabel=ylabel,
        alabel="E/(V/m)",
        figsize=(10, 8),
        title=title,
        ret_ax=ret_ax,
    )


def laser_to_openPMD(
    laser,
    file_prefix,
    Nt=None,
    Nx=None,
    Ny=None,
    write_dir="diags",
    file_format="bp",
    iteration=0,
    points_between_r=1,
    cut_first_frac=0,
    forced_dt=None,
    offset_frac=0,
    data_step=1,
    conversion_safety=1.1,
    append=False,
    show=False,
):
    """
    Write the laser field into an openPMD file.

    Parameters
    ----------
    laser : lasy Laser object
        The laser to be written into the openPMD-file. If it is dim='rt' it is first turned into an 'xyt'-laser.

    Nt : int (optional)
        Number of time points on which field should be sampled. If is None,
        the orignal time grid is used, otherwise field is interpolated on a
        new grid.

    Nx : int (optional)
        Number of x-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    Ny : int (optional)
        Number of y-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    file_prefix : string
        The file name will start with this prefix.

    write_dir : string (optional)
        The directory where the file will be written.

    file_format : string (optional)
        Format to be used for the output file. Options are "h5" and "bp".

    iteration : int (optional)
        The iteration number for the file to be written.

    points_between_r : int or float (optional)
        if laser.dim=="rt" the field is converted to xyt to write into the file. T
        his argument describes, how many points in x and y directions should be placed
        (interpolated) between two given values in the r direction.

    cut_first_frac : float between 0 and 1 (optional)
        fraction of the points in the t-direction to be cut

    forced_dt : float (optional)
        forces dt to be this value, if possible.
        if forced_dt and Nt is given the section is taken from the end of the field.

    offset_frac : float between 0 and 1 (optional)
        only relevant, when both Nt and forced_dt are given. offsets the Nt * forced_dt
        range from the upper end. This is interpreted as the fraction of the original field
        the offset is supposed to be.

    data_step : int (optional)
        Only saves every (data_step)th data point to the file transversally.
        The number Nx is still reached.

    conversion_safety : float (optional)
        Only relevant, when laser is 'rt'. Gives the safety factor for the conversion to 'xyt'.

    append : bool (optional)
        needs to be set to True, if the file should be added to not overwritten.
    """
    print("preparing the laser...")

    # 1. Check if cutting is neccessary
    changed = False
    if cut_first_frac > 0:
        laser_new = _cut_first_frac(laser, cut_first_frac)
        changed = True

    # 2. If the laser is in rt it needs to be transformed to xyt, because we want to save a 3D electric field. Assuming cylindrical symmetry in this case.
    if laser.dim == "rt":
        if Nx is None or Ny is None:
            raise ValueError("If given a laser in dim='rt' both Nx and Ny must be given")
        if changed:
            laser_new = _rt_to_xyt(
                laser_new,
                int(conversion_safety * Nx * data_step),
                int(conversion_safety * Ny * data_step),
                points_between_r=points_between_r,
            )
        else:
            laser_new = _rt_to_xyt(
                laser,
                int(conversion_safety * Nx * data_step),
                int(conversion_safety * Ny * data_step),
                points_between_r=points_between_r,
            )
        changed = True
    if not changed:
        laser_new = laser

    print("extracting full field...")
    if Nx is not None:
        Nx *= data_step
    if Ny is not None:
        Ny *= data_step

    # 3. Actually extracting the full electric field.
    f, ext = get_full_field(laser_new, Nt=Nt, Nx=Nx, Ny=Ny, forced_dt=forced_dt, offset_frac=offset_frac)

    print("saving...")

    # 4. Save to file
    write_to_openpmd_file(
        write_dir,
        file_prefix,
        file_format,
        f,
        ext,
        laser_new.profile.lambda0,
        laser_new.profile.pol,
        iteration=iteration,
        data_step=data_step,
        append=append,
    )

    # 5. possibly show the saved field to the screen
    if show:
        _show(f, ext, True, 0.01, "Saved field", False)
    print("done")


def memory_calc(Nx, Ny, Nt):
    """calculates the meory in bytes required by a file generated by the
    laser_to_openPMD function.

    Parameters:
    Nx, Ny, Nt : int
        The number of points in the array. Should be the same as for the function call.
    """
    return Nx * Ny * Nt * 32 / 8


def show_field(
    laser,
    Nt=None,
    Nx=None,
    Ny=None,
    Nr=None,
    forced_dt=None,
    offset_frac=0,
    show_SI=True,
    linthresh_frac=0.01,
    title="",
    ret_ax=False,
):
    """shows the laser field on an symlog-plot.

    Parameters:
    laser : lasy Laser object
        The laser to be shown.

    Nt : int (optional)
        Number of time points on which field should be sampled. If is None,
        the orignal time grid is used, otherwise field is interpolated on a
        new grid.

    Nx : int (optional)
        Number of x-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    Ny : int (optional)
        Number of y-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    Nr : int (optional)
        Number of r-points the field should be cut dow to. Does not interpolate.
        Always in the middle.

    forced_dt : float (optional)
        forces dt to be this value, if possible.
        if forced_dt and Nt is given the section is taken from the end of the field.

    offset_frac : float between 0 and 1 (optional)
        only relevant, when both Nt and forced_dt are given. offsets the Nt * forced_dt
        range from the upper end. This is interpreted as the fraction of the original field
        the offset is supposed to be.

    linthresh_frac : float (optional)
        fraction of the absolute maximum from where the normalistion should be linear.

    show_SI : bool (optional)
        If True the axes are written in SI-units, otherwise they are written in points.

    title : str (optional)
        The title of the plot.

    ret_ax : bool (optional)
        if True the figure object and the axes object are returned, otherwise displayed.

    Returns:
    fig : pyplot figure object (optional)
        only returned if ret_ax is True

    ax : pyplot axes object (optional)
        only returned if ret_ax is True
    """
    print("Extracting full field")
    if Ny is None:
        Ny = 2
    f, ext = get_full_field(laser, Nt=Nt, Nr=Nr, Nx=Nx, Ny=Ny, forced_dt=forced_dt, offset_frac=offset_frac)
    print("Displaying")
    return _show(f, ext, show_SI, linthresh_frac, title, ret_ax)
