import openpmd_api as api
import numpy as np
import math


def shape_swap(shape, side_str):
    str_to_side = {'x': (0, 0), 'xr': (0, 1),
                   'y': (1, 0), 'yr': (1, 1),
                   'z': (2, 0), 'zr': (2, 1)}
    side_tuple = str_to_side[side_str]
    axes_maps = ((2, 1, 0), (2, 0, 1), (1, 0, 2))
    axes_map = axes_maps[side_tuple[0]]
    new_shape = [0, 0, 0]
    for ii in range(3):
        new_shape[ii] = shape[axes_map[ii]]
    return tuple(new_shape)


def axis_swap(data, side_str):
    str_to_side = {'x': (0, 0), 'xr': (0, 1),
                   'y': (1, 0), 'yr': (1, 1),
                   'z': (2, 0), 'zr': (2, 1)}
    side_tuple = str_to_side[side_str]
    axes_maps = ((2, 1, 0), (2, 0, 1), (1, 0, 2))
    data = np.moveaxis(data, axes_maps[side_tuple[0]], (0, 1, 2))
    reverses = [[(True, False, False), (True, True, True)],
                [(True, True, False), (True, False, True)],
                [(True, False, False), (True, True, True)]]
    reverse = reverses[side_tuple[0]][side_tuple[1]]
    axes_to_flip = []
    for axis, flip in enumerate(reverse):
        if flip:
            axes_to_flip.append(axis)
    if axes_to_flip:
        data = np.flip(data, axis=axes_to_flip)
    return data


def second_point(first_point, yaw, pitch):
    param = 1
    x1, y1, z1 = first_point
    x2 = x1 + param * math.sin(pitch)
    y2 = y1 - param * math.cos(pitch) * math.sin(yaw)
    z2 = z1 + param * math.cos(pitch) * math.cos(yaw)
    return np.array([x2, y2, z2])


def norm2(x):
    return math.sqrt((x ** 2).sum())


def gaussian(dist, sigma):
    tmp = dist / sigma
    exponent = -0.5 * (tmp * tmp)
    return math.exp(exponent)


def distance(point, first_line_point, second_line_point):
    numerator = norm2(
        np.cross((point - first_line_point), (point - second_line_point)))
    denominator = norm2(second_line_point - first_line_point)
    return numerator / denominator


def generate_intensity_array(shape, start_point, yaw, pitch, sigma,
                             grid_spacing):
    second_line_point = second_point(start_point, yaw, pitch)

    output = np.empty(shape)
    with np.nditer(output, flags=['multi_index'],
                   op_flags=['writeonly']) as it:
        for cell in it:
            point = np.array(it.multi_index, dtype=np.float64) + np.array(
                [0.5, 0.5, 0.5])
            point *= grid_spacing
            dist = distance(point, start_point, second_line_point)
            cell[...] = gaussian(dist, sigma)
    return output


def verify_results(picongpu_run_dir, sigma, side_str, shape, yaw,
                   pitch, offset):
    # load simulation output
    infix = "_%06T"
    backend = "h5"
    mesh_name = "externalBeamIntensity"
    openpmd_path = picongpu_run_dir / 'debugExternalBeam'
    series_path_str = "{}/{}{}.{}".format(str(openpmd_path),
                                          'debugExternalBeam',
                                          infix, backend)
    series = api.Series(series_path_str, api.Access.read_only)

    checks = [[], []]
    shape_det = shape_swap(shape, side_str)
    for iteration in series.read_iterations():
        mesh = iteration.meshes[mesh_name]
        mrc = mesh[api.Mesh_Record_Component.SCALAR]
        data = mrc.load_chunk()
        series.flush()
        data *= mrc.unit_SI
        data = np.moveaxis(data, (2, 1, 0), (0, 1, 2))
        data = axis_swap(data, side_str)
        start_point = np.array([0.5, 0.5, 0]) * np.array(data.shape)
        start_point[0] += offset[0]
        start_point[1] += offset[1]
        grid_spacing = np.array(mesh.grid_spacing) * np.array(
            mesh.grid_unit_SI)
        start_point *= grid_spacing
        intensity = generate_intensity_array(shape_det, start_point, yaw,
                                             pitch,
                                             sigma, grid_spacing)
        check = np.all(np.isclose(data, intensity))
        np.set_printoptions(threshold=100)
        # print("data: ", data)
        # print('intensity: ', intensity)
        checks[0].append(check)
        checks[1].append(iteration.iteration_index)
    return checks
