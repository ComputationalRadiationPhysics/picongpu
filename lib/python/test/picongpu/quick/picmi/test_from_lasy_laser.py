"""
This file is part of PIConGPU.

Copyright 2025-2026 Edgar Marquardt

Test the FromLasyLaser. It is not a unit test, but a quick test to see if the basic functionality works.
"""

from picongpu import picmi

from lasy.laser import Laser
from lasy.profiles import GaussianProfile

import openpmd_api as io
import numpy as np
from scipy.constants import c, epsilon_0

import tempfile
import os

import matplotlib.pyplot as plt

# test params
xyt_lo = (-1e-3, -1e-3, -1e-13)
rt_lo = (0, -1e-13)
xyt_hi = (1e-3, 1e-3, 1e-13)
rt_hi = (2e-3, 1e-13)
xyt_npoints = (60, 60, 12)
rt_npoints = (60, 12)
lambda_0 = 7e-7
E_laser = 1e-2
w0 = 5e-4
tau0 = 1e-13


# helper functions
def read_file(
    directory, filename, iteration=0, direction="x", cleanup=True, reshape_npoints=xyt_npoints, ret_ext=False
):
    """reads the electric field in the file at the path filepath and returns it as an array."""
    series = io.Series(os.path.join(directory.name, filename), io.Access.read_only)
    E = series.iterations[iteration].meshes["E"][direction][:, :, :]
    series.flush()
    if ret_ext:
        spacing = series.iterations[iteration].meshes["E"].get_attribute("gridSpacing")
        spacing[0] /= c
        lo = tuple([-(E.shape[i] - 1) / 2 * spacing[i] for i in range(3)][::-1])
        hi = tuple([(E.shape[i] - 1) / 2 * spacing[i] for i in range(3)][::-1])
        ext = [lo, hi]
    series.close()
    if cleanup:
        directory.cleanup()
    if ret_ext:
        return np.transpose(E), ext
    return np.transpose(E)


def get_grid_axes(dim, Nt=None):
    """returns the axes of the grid following the test params"""
    if dim == "xyt":
        if Nt is not None:
            npoints = (xyt_npoints[0], xyt_npoints[1], Nt)
        else:
            npoints = xyt_npoints
        axes = [np.linspace(xyt_lo[i], xyt_hi[i], npoints[i]) for i in range(3)]
        mesh_axes = np.meshgrid(axes[1], axes[0], axes[2])
    elif dim == "rt":
        if Nt is not None:
            npoints = (rt_npoints[0], Nt)
        else:
            npoints = rt_npoints
        axes = [np.linspace(rt_lo[i], rt_hi[i], npoints[i]) for i in range(2)]
        axes[0] = np.hstack((axes[0][::-1][:-1], axes[0]))
        mesh_axes = np.meshgrid(axes[0], axes[1])
        mesh_axes = list(map(np.transpose, mesh_axes))
    else:
        raise ValueError("What is dim " + dim + " supposed to mean?")
    return mesh_axes


def grid_cell_volume(dim, Nt=None):
    """calculates the volume each point on the grid describes"""
    if dim == "xyt":
        dx = (xyt_hi[0] - xyt_lo[0]) / (xyt_npoints[0] - 1)
        dy = (xyt_hi[1] - xyt_lo[1]) / (xyt_npoints[1] - 1)
        if Nt is None:
            dt = (xyt_hi[-1] - xyt_lo[-1]) / (xyt_npoints[-1] - 1)
        else:
            dt = (xyt_hi[-1] - xyt_lo[-1]) / (Nt - 1)
        return dx * dy * dt * c

    else:
        r, dr = np.linspace(-rt_hi[0], rt_hi[0], 2 * rt_npoints[0] - 1, retstep=True)
        if Nt is None:
            dt = (rt_hi[-1] - rt_lo[-1]) / (rt_npoints[-1] - 1)
        else:
            dt = (rt_hi[-1] - rt_lo[-1]) / (Nt - 1)
        dV = np.pi * ((abs(r) + dr / 2) ** 2 - (abs(r) - dr / 2) ** 2) * dt * c / 2
        # dV[len(dV)//2] = np.pi * (dr/2) ** 2 * dt * c
        # That should be neccessary but Lasy does not do it so I dont either.
        return dV[:, None]


def scale_to_energy(dim, field, energy, Nt=None):
    """returns the same field but scaled to have the given energy"""
    dV = grid_cell_volume(dim, Nt=Nt)
    # np.abs(field) ** 2 is already the complete energy density of the laser field,
    # because the B-field is found in the imaginary component of the field in this complex notation.
    # Of course, this assumes only one direction of the E-field and one direction of the B-field being > 0.
    E_curr = np.sum(np.abs(field) ** 2 * dV) * epsilon_0 / 2
    scale = np.sqrt(energy / E_curr)

    return field * scale


def gaussian_pulse(dim, Nt=None):
    """returns the expected field of the gaussian pulse using the test params"""
    axes = get_grid_axes(dim, Nt=Nt)
    t = axes[-1]

    t_gauss = np.exp(-t * t / (tau0 * tau0) - 2j * np.pi * t * c / lambda_0)
    if dim == "xyt":
        x = axes[0]
        y = axes[1]
        tr_gauss = np.exp(-x * x / (w0 * w0) - y * y / (w0 * w0))
    else:
        r = axes[0]
        tr_gauss = np.exp(-r * r / (w0 * w0))

    E = np.real(scale_to_energy(dim, t_gauss * tr_gauss, E_laser, Nt=Nt))

    return E


def show(field, marks=["", "", ""], **kwargs):
    """display the contents of field graphically"""
    fig = plt.figure()
    ax = fig.add_subplot()
    if len(field.shape) > 2:
        field = field[field.shape[0] // 2, :, :]
    img = ax.imshow(field, **kwargs)
    clb = plt.colorbar(img)
    ax.set_xlabel(marks[0])
    ax.set_ylabel(marks[1])
    clb.set_label(marks[2])
    if len(marks) > 3:
        ax.set_title(marks[3])
    plt.show()


def double_inputs(lo, hi, npoints):
    dx = (hi[0] - lo[0]) / (npoints[0] - 1)
    dy = (hi[1] - lo[1]) / (npoints[1] - 1)
    # this is neccessary, because for N points there are only N-1 steps, that need to be covered by lo and hi.
    # Therefore when we double N we need to add half a step to lo and hi in addition to doubling them.
    lo = (lo[0] * 2 - dx / 2, lo[1] * 2 - dy / 2, lo[2])
    hi = (hi[0] * 2 + dx / 2, hi[1] * 2 + dy / 2, hi[2])
    npoints = (npoints[0] * 2, npoints[1] * 2, npoints[2])
    return lo, hi, npoints


def assert_isclose(actual, desired, rtol=1e-7):
    """determines, whether the fields actual (using Lasy and pll) and desired (using gaussian_pulse) are close with a maximum
    relative error below rtol. Otherwise display the fields for further investigation."""
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol)
    except AssertionError as error:
        assert actual.shape == desired.shape
        if len(actual.shape) == 3:
            print("A sample of the mismatching fields at [:, :, actual.shape[2] // 2]:")
            show(np.abs(actual)[:, :, actual.shape[2] // 2], marks=["x", "y", "", "E_lasy"], norm="log")
            show(np.abs(desired)[:, :, desired.shape[2] // 2], marks=["x", "y", "", "E_gauss"], norm="log")
            show(
                np.abs((actual - desired) / desired)[:, :, desired.shape[2] // 2],
                marks=["x", "y", "", "relative error"],
                norm="log",
            )
        else:
            print("The mismatching fields:")
            show(np.abs(actual), marks=["t", "r", "", "E_lasy"], norm="log")
            show(np.abs(desired), marks=["t", "r", "", "E_gauss"], norm="log")
            show(np.abs((actual - desired) / desired), marks=["t", "r", "", "relative error"], norm="log")
        raise error


def picmi_generate_input(directory, laser, **kwargs):
    """generates the input files for a (theoretical) PIConGPU simulation in the
    given directory, including the laser as incident field.
    """
    boundary_conditions = ["periodic", "periodic", "periodic"]
    grid = picmi.Cartesian3DGrid(
        number_of_cells=xyt_npoints,
        lower_bound=xyt_lo,
        upper_bound=xyt_hi,
        lower_boundary_conditions=boundary_conditions,
        upper_boundary_conditions=boundary_conditions,
    )
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    laser = picmi.FromLasyLaser(
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        time_offset_si=0.0,
        file_path=os.path.join(directory.name, "test_laser.bp"),
        lasyLaser=laser,
        **kwargs,
    )
    sim = picmi.Simulation(
        time_step_size=9.65531e-14,
        max_steps=1024,
        solver=solver,
    )
    sim.add_laser(laser, None)
    sim.write_input_file(os.path.join(directory.name, "setup/"))


# test functions
def test_laser_file():
    """test whether the laser file is generated correctly"""
    laser = Laser("xyt", xyt_lo, xyt_hi, xyt_npoints, GaussianProfile(lambda_0, (1, 0), E_laser, w0, tau0, 0))

    td = tempfile.TemporaryDirectory()
    picmi_generate_input(td, laser)

    E = read_file(td, "test_laser.bp")
    assert_isclose(E, gaussian_pulse("xyt"))


def test_laser_file_rt_to_xyt():
    """test whether the laser file is generated correctly with rt->xyt conversion"""
    laser = Laser("rt", rt_lo, rt_hi, rt_npoints, GaussianProfile(lambda_0, (1, 0), E_laser, w0, tau0, 0))

    td = tempfile.TemporaryDirectory()
    picmi_generate_input(td, laser, Nx=xyt_npoints[0], Ny=xyt_npoints[1])

    E = read_file(td, "test_laser.bp")
    # tolerance because of linear interpolation. it can not get much better.
    assert_isclose(E, gaussian_pulse("xyt"), rtol=2e-2)


def test_laser_file_t_interpolate():
    """test whether the laser file is generated correctly with time interpolation"""
    laser = Laser("xyt", xyt_lo, xyt_hi, xyt_npoints, GaussianProfile(lambda_0, (1, 0), E_laser, w0, tau0, 0))

    td = tempfile.TemporaryDirectory()
    picmi_generate_input(td, laser, Nt=20)

    E = read_file(td, "test_laser.bp", reshape_npoints=(xyt_npoints[0], xyt_npoints[1], 20))
    # tolerance because of linear interpolation. it can not get much better.
    assert_isclose(E, gaussian_pulse("xyt", Nt=20), rtol=2e-2)


def test_input_files():
    """test whether the incidentField.param is generated correctly"""
    laser = Laser("xyt", xyt_lo, xyt_hi, xyt_npoints, GaussianProfile(lambda_0, (1, 0), E_laser, w0, tau0, 0))

    td = tempfile.TemporaryDirectory()
    picmi_generate_input(td, laser)
    print(td.name)
