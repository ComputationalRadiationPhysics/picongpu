"""
This file is part of the PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""
import openpmd_api as api
import numpy as np
import math
import json
import argparse
from pathlib import Path
import scipy.constants as cs


def get_beam_unit_vectors(side_str):
    """
    Returns beam coordinate system unit vectors for given side as defined
    in Side.hpp in PIConGPU.

    :param side_str: String defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
    :return: beam coordinate system unit vectors (x,y,z)
    """
    str_to_z_dir = {'x': (1, 0, 0), 'xr': (-1, 0, 0), 'y': (0, 1, 0),
                    'yr': (0, -1, 0), 'z': (0, 0, 1), 'zr': (0, 0, -1)}

    str_to_y_dir = {'x': (0, 1, 0), 'xr': (0, -1, 0), 'y': (-1, 0, 0),
                    'yr': (1, 0, 0), 'z': (-1, 0, 0), 'zr': (-1, 0, 0)}

    str_to_x_dir = {'x': (0, 0, -1), 'xr': (0, 0, -1), 'y': (0, 0, -1),
                    'yr': (0, 0, -1), 'z': (0, 1, 0), 'zr': (0, -1, 0)}
    # normalize
    z = np.array(str_to_z_dir[side_str])
    z = z / np.linalg.norm(z)
    x = np.array(str_to_x_dir[side_str])
    x = x / np.linalg.norm(x)
    y = np.array(str_to_y_dir[side_str])
    y = y / np.linalg.norm(y)
    # check orthogonality
    assert np.dot(x, y) == 0
    assert np.dot(x, z) == 0
    # check right-handedness
    assert np.dot(np.cross(x, y), z) > 0
    return x, y, z


def get_beam_origin(side_str, sim_box_size, offset_pic):
    """ Get beam coordinate system origin in PIConGPU system

    :param side_str: String defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
    :param sim_box_size: Size of the simulation box (x, y, z) tuple.
    :param offset_pic: Beam origin offset in PIConGPU coordinates
    :return: origin position (x, y, z)
    """
    str_to_origin = {'x': (0, 0.5, 0.5), 'xr': (1.0, 0.5, 0.5),
                     'y': (0.5, 0, 0.5), 'yr': (0.5, 1, 0.5),
                     'z': (0.5, 0.5, 0), 'zr': (0.5, 0.5, 1)}
    return sim_box_size * np.array(str_to_origin[side_str]) + offset_pic


def transfer_coordinates(v, x_hat, y_hat, z_hat):
    """ Transform a vector into a new coordinate system

    :param v: vector to transfer
    :param x_hat: x base vector of the new coordinate system
    :param y_hat: y base vector of the new coordinate system
    :param z_hat: z base vector of the new coordinate system
    :return:
    """
    return np.array((np.dot(v, x_hat), np.dot(v, y_hat), np.dot(v, z_hat)))


def get_beam_coordinate_system(side_str, offset_beam, sim_box_size):
    """ Get beam coordinate system

    :param side_str: String defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
    :param offset_beam: Beam transversal offset in beam coordinate system (x,y)
    :param sim_box_size: Size of the simulation box (x, y, z) tuple.
    :return: beam system origin, x base vector, y base vector, z base vector
        in PIConGPU coordinates
    """
    x_beam, y_beam, z_beam = get_beam_unit_vectors(side_str)
    x_pic = transfer_coordinates(np.array((1, 0, 0)), x_beam, y_beam, z_beam)
    y_pic = transfer_coordinates(np.array((0, 1, 0)), x_beam, y_beam, z_beam)
    z_pic = transfer_coordinates(np.array((0, 0, 1)), x_beam, y_beam, z_beam)
    offset_beam = np.array([offset_beam[0], offset_beam[1], 0])
    offset_pic = transfer_coordinates(offset_beam, x_pic, y_pic, z_pic)
    origin = get_beam_origin(side_str, sim_box_size, offset_pic)
    return origin, x_beam, y_beam, z_beam


class BeamCoordinates:
    """
    Functor converting a vector in the PIConGPU coordinate system into
    the beam coordinate system.
    """

    def __init__(self, side_str, offset_beam, sim_box_size):
        """Constructor

        :param side_str: String defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
        :param offset_beam: Beam transversal offset in beam coordinate system
            (x,y)
        :param sim_box_size: Size of the simulation box (x, y, z) tuple.
        """
        self.origin, self.x_beam, self.y_beam, self.z_beam = \
            get_beam_coordinate_system(side_str,
                                       offset_beam,
                                       sim_box_size)

    def __call__(self, x, y, z):
        """ Functor implementation

        :param x: x coordinate in PIConGPU system
        :param y: y coordinate in PIConGPU system
        :param z: z coordinate in PIConGPU system
        :return: x, y, z in beam coordinates
        """
        x -= self.origin[0]
        y -= self.origin[1]
        z -= self.origin[2]
        x_b = x * self.x_beam[0] + y * self.x_beam[1] + z * self.x_beam[2]
        y_b = x * self.y_beam[0] + y * self.y_beam[1] + z * self.y_beam[2]
        z_b = x * self.z_beam[0] + y * self.z_beam[1] + z * self.z_beam[2]
        return x_b, y_b, z_b


def gaussian(x, sigma):
    """gaussian function"""
    tmp = x / sigma
    exponent = -0.5 * (tmp * tmp)
    return math.exp(exponent)


class GaussianProfile:
    """Gaussian beam profile"""

    def __init__(self, sigma_x, sigma_y):
        """Constructor

        :param sigma_x: sigma in x direction
        :param sigma_y: sigma in y direction
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, x, y):
        """ Functor implementation

        :param x: evaluation position x coordinate
        :param y: evaluation position y coordinate
        :return: intensity factor at the given point (normalized to max=1)
        """
        x_term = (x / self.sigma_x) ** 2
        y_term = (y / self.sigma_y) ** 2
        exponent = -0.5 * (x_term + y_term)
        return np.exp(exponent)


class ConstShape:
    """Constant beam temporal shape"""

    def __init__(self, start_time, end_time):
        """Constructor

            :param start_time: beam start time
            :param end_time: time at which the beam ends
            """
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, t):
        """Functor implementation

            :param t: time, numpy array
            :return: intensity factor normalized to max=1
            """
        mask = np.where((t >= self.start_time) & (t < self.end_time))
        factor = np.zeros_like(t)
        factor[mask] = 1
        return factor


def generate_intensity_array(sim_shape,
                             beam_profile,
                             beam_shape,
                             coor_trans,
                             delta_x,
                             delta_y,
                             delta_z,
                             t):
    """ Generate intensity array for comparison

    :param sim_shape: simulation shape in cells (x, y, z)
    :param beam_profile: a callable defining the beam transversal profile
        takes x and y as positional arguments
    :param beam_shape:  a callable defining the beam temporal shape
        takes time as positional argument
    :param coor_trans: a callable defining the coordinate transform to the beam
        coordinate system. Takes x,y, and z as positional arguments.
    :param delta_x: cell size in x
    :param delta_y: cell size in y
    :param delta_z: cell size in z
    :param t: time
    :return: returns the 3D intensity array normed to max=1
    """
    x_pic = np.arange(sim_shape[0]) + 0.5
    y_pic = np.arange(sim_shape[1]) + 0.5
    z_pic = np.arange(sim_shape[2]) + 0.5

    x_pic *= delta_x
    y_pic *= delta_y
    z_pic *= delta_z
    x_beam, y_beam, z_beam = coor_trans(x_pic[:, None, None],
                                        y_pic[None, :, None],
                                        z_pic[None, None, :])
    return beam_profile(x_beam, y_beam) * beam_shape(t - z_beam / cs.c)


def verify_results(picongpu_run_dir, side_str, offset):
    """ Verify an ExternalBeam test run

    :param picongpu_run_dir: run directory, pathlib Path object
    :param side_str: String defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
    :param offset: offset in cells from the default beam center position
        in the transversal beam plane. (x,y) in beam coordinate system.
    :return: list of two lists, first list contains test results for all
        iterations (true or false value), the second list contains interation
        numbers.
    """
    time_step = 1.0e-6 / cs.c
    cell_size = 3.0e-6
    sigma = (10, 7)
    start_time = 0  # time_steps
    end_time = 20  # time_steps
    profile = GaussianProfile(sigma[0] * cell_size, sigma[1] * cell_size)
    shape = ConstShape(time_step * start_time, time_step * end_time)
    sim_shape = (24, 24, 24)
    coor_trans = BeamCoordinates(side_str, (offset[0] * cell_size,
                                            offset[1] * cell_size),
                                 np.array(sim_shape) * cell_size)

    factor = 1e5  # max photon count per cell

    # load simulation output
    infix = "_%06T"
    backend = "bp"
    mesh_name = "ph_all_particleCounter"
    openpmd_path = picongpu_run_dir / 'openPMD'
    series_path_str = "{}/{}{}.{}".format(str(openpmd_path),
                                          'simData',
                                          infix,
                                          backend)
    series = api.Series(series_path_str, api.Access.read_only)

    checks = [[], []]
    for iteration in series.read_iterations():
        mesh = iteration.meshes[mesh_name]
        mrc = mesh[api.Mesh_Record_Component.SCALAR]
        data = mrc.load_chunk()
        series.flush()
        data *= mrc.unit_SI
        # we work with PIConGPU coordinates: x,y,z
        # so that the openPMD array needs to be rearranged first
        data = data.T  # z,y,x -> x,y,z
        axis_map = {'x': 0, 'xr': 0, 'y': 1, 'yr': 1, 'z': 2, 'zr': 2}
        reverse_map = {'x': False, 'xr': True, 'y': False, 'yr': True,
                       'z': False, 'zr': True}
        prop_axis_idx = axis_map[side_str]

        # light travels 1/3 of a cell in one time step, so we fill-up only
        # 1/3 of a cell on an iteration. Here we compute the intensity for an
        # array of such 1/3 sub-cells and calculate the particle number per
        # cell with the simulation resolution later by summing over the
        # sub-cells.
        reduced_cells_shape = list(sim_shape)
        reduced_cells_shape[prop_axis_idx] *= 3
        reduced_cells_shape = tuple(reduced_cells_shape)

        reduced_cell_size = [cell_size, cell_size, cell_size]
        reduced_cell_size[prop_axis_idx] /= 3
        reduced_cell_size = tuple(reduced_cell_size)

        particle_count = generate_intensity_array(reduced_cells_shape,
                                                  profile,
                                                  shape,
                                                  coor_trans,
                                                  *reduced_cell_size,
                                                  (iteration.iteration_index
                                                   - 1) * time_step)

        class SwapIdx:
            """Converts a slicing object by swapping slices for two indices"""
            def __init__(self, idx1, idx2):
                """Constructor

                :param idx1: first index to swap
                :param idx2: second index to swap
                """
                self.idx1 = idx1
                self.idx2 = idx2

            def __getitem__(self, slicing):
                """ Square brackets operator implementation

                :param slicing: slicing such as e.g. [1:1, :, ::2]
                :return: the slicing with swapped axes
                """
                # make input mutable
                slicing = list(slicing)
                # swap slices
                slicing[self.idx1], slicing[self.idx2] = slicing[self.idx2], \
                    slicing[self.idx1]
                # slicing is a tuple of slice objects
                return tuple(slicing)

        # the rest is written with the propagation axis being the first axis
        # we use swap to include all possible orientations
        swap = SwapIdx(0, prop_axis_idx)

        if reverse_map[side_str]:
            particle_count[swap[:-1, :, :]] = particle_count[swap[1:, :, :]]
            particle_count[swap[-1, :, :]] = 0.0
        else:
            particle_count[swap[1:, :, :]] = particle_count[swap[:-1, :, :]]
            particle_count[swap[0, :, :]] = 0.0

        # summ-up over sub-cells
        particle_count = particle_count[swap[0::3, :, :]] / 3 + particle_count[
            swap[1::3, :, :]] / 3 + particle_count[swap[2::3, :, :]] / 3
        # multiply th normalized array with the expected max particle count
        # per cell
        particle_count *= factor
        check = np.all(np.isclose(data, particle_count))
        checks[0].append(check)
        checks[1].append(iteration.iteration_index)
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify an ExternalBeam test")
    parser.add_argument('--dir',
                        nargs='?',
                        help="the simOutput directory",
                        default=Path.cwd())
    args = parser.parse_args()
    picongpu_run_dir = Path(args.dir)
    with open(picongpu_run_dir / "test_setup.json", "r") as f:
        parameters = json.load(f)
    checks = verify_results(picongpu_run_dir, **parameters)
    test_passed = np.all(checks[0])
    print(fr"Test result: {test_passed}")
    if not test_passed:
        for ii, iteration in enumerate(checks[1]):
            print(fr"iteration: {iteration}, "
                  fr"{'passed' if checks[0][ii] else 'failed'} \n")
    assert np.all(checks[0])
