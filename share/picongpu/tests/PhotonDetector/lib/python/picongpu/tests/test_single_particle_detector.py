import os
import pytest
import subprocess
import openpmd_api as api
from pathlib import Path
import numpy as np


# requires pytest-subtests


def rotate(vec, placement):
    """

    :param vec:
    :param placement:
    :return:
    """
    placement_to_side = {'x-': (0, 1), 'x+': (0, 0),
                         'y-': (1, 1), 'y+': (1, 0),
                         'z-': (2, 1), 'z+': (2, 0)}
    side = placement_to_side[placement]

    axes_maps = ((2, 1, 0), (2, 0, 1), (1, 0, 2))
    reverses = [[(True, False, False), (True, True, True)],
                [(True, True, False), (True, False, True)],
                [(True, False, False), (True, True, True)]]

    axes_map = axes_maps[side[0]]
    reverse = reverses[side[0]][side[1]]
    result = 3 * [None]
    result[axes_map[0]] = vec[0] * (-1.0 if reverse[0] else 1.0)
    result[axes_map[1]] = vec[1] * (-1.0 if reverse[1] else 1.0)
    result[axes_map[2]] = vec[2] * (-1.0 if reverse[2] else 1.0)
    return result


@pytest.mark.parametrize(
    'accum_policy,placement,cell_to_hit_det,backend,infix',
    [  # test all cells for one orientation
        ("CountParticles", "y+", (1, -1), "json", "NULL"),
        ("CountParticles", "y+", (1, 0), "json", "NULL"),
        ("CountParticles", "y+", (1, 1), "json", "NULL"),
        ("CountParticles", "y+", (0, -1), "json", "NULL"),
        ("CountParticles", "y+", (0, 0), "json", "NULL"),
        ("CountParticles", "y+", (0, 1), "json", "NULL"),
        ("CountParticles", "y+", (-1, -1), "json", "NULL"),
        ("CountParticles", "y+", (-1, 0), "json", "NULL"),
        ("CountParticles", "y+", (-1, 1), "json", "NULL"),
        # test all other placements for one cell
        # (1,-1) on detector so no zeros
        #  -> not a special case
        ("CountParticles", "y-", (1, -1), "json", "NULL"),
        ("CountParticles", "x-", (1, -1), "json", "NULL"),
        ("CountParticles", "x+", (1, -1), "json", "NULL"),
        ("CountParticles", "z-", (1, -1), "json", "NULL"),
        ("CountParticles", "z+", (1, -1), "json", "NULL"),
        # check the other accumulation policy
        ("AddWaveParticles", "x-", (1, -1), "json", "NULL"),
        # try out different backends for one configuration
        ("CountParticles", "x-", (1, -1), "json", "_%06T"),
        # fails, gh issue: https://github.com/openPMD/openPMD-api/issues/996
        # ("CountParticles", "x-", (1, -1), "bp", "NULL"),
        ("CountParticles", "x-", (1, -1), "bp", "_%06T"),
        ("CountParticles", "x-", (1, -1), "h5", "NULL"),
        ("CountParticles", "x-", (1, -1), "bp", "_%06T"),
    ])
def test_detect_single_particle(tmp_path, accum_policy, placement,
                                cell_to_hit_det, backend, infix):
    # use pic create to create a tmp example dir
    example_root = tmp_path / "example"
    example_src = Path(os.environ.get(
        "PICSRC")) / 'share' / 'picongpu' / 'tests' / 'PhotonDetector'

    create_result = subprocess.run(
        ["pic-create", f"{str(example_src.resolve())}",
         f"{str(example_root.resolve())}"], check=True)
    assert create_result.returncode == 0

    # some simulation params
    steps = 115
    grid_size = 64
    detector_period = f"{steps}:{steps}"
    det_to_array = {(+1, -1): (+0, +0), (+1, +0): (+0, +1), (+1, +1): (+0, +2),
                    (+0, -1): (+1, +0), (+0, +0): (+1, +1), (+0, +1): (+1, +2),
                    (-1, -1): (+2, +0), (-1, +0): (+2, +1), (-1, +1): (+2, +2)}
    cell_to_hit = det_to_array[cell_to_hit_det]
    # prepare  cmake flags
    a = 0.2 / 5.0
    orientations_det = [[(+a, -a), (+a, +0), (+a, +a)],
                        [(+0, -a), (+0, +0), (+0, +a)],
                        [(-a, -a), (-a, +0), (-a, +a)]]
    orientation_det = orientations_det[cell_to_hit[0]][cell_to_hit[1]]
    orientation = rotate((orientation_det[0], orientation_det[1], 1),
                         placement)
    b = grid_size // 2
    c = grid_size - 1
    placement_to_cell = {'x+': (0, b, b), 'x-': (c, b, b),
                         'y+': (b, 0, b), 'y-': (b, c, b),
                         'z+': (b, b, 0), 'z-': (b, b, c)}
    photon_placement = placement_to_cell[placement]

    overwrite_list = f"-DPARAM_ACCUMULATION_POLICY={accum_policy};" \
                     f"-DPARAM_PHOTON_DIRECTION_X={orientation[0]};" \
                     f"-DPARAM_PHOTON_DIRECTION_Y={orientation[1]};" \
                     f"-DPARAM_PHOTON_DIRECTION_Z={orientation[2]};" \
                     f"-DPARAM_PHOTON_PLACEMENT_X={photon_placement[0]}u;" \
                     f"-DPARAM_PHOTON_PLACEMENT_Y={photon_placement[1]}u;" \
                     f"-DPARAM_PHOTON_PLACEMENT_Z={photon_placement[2]}u;"
    cmake_flags = f"-DPARAM_OVERWRITES:LIST='{overwrite_list}'"
    print(cmake_flags)

    # compile
    compile_command = ["pic-build", "-c", f"\"{cmake_flags}\""]
    compile_result = subprocess.run(compile_command, cwd=example_root)
    # continue only after a successful compilation
    assert compile_result.returncode == 0

    picongpu_run_dir = example_root / "simOutput"
    picongpu_run_dir.mkdir()
    picongpu_exec = example_root / 'bin' / 'picongpu'

    picongpu_command = ["mpiexec", "-n", "1",
                        f"{picongpu_exec}",
                        "-d", "1", "1", "1",
                        "-s", f"{steps}",
                        "-g", f"{grid_size}", f"{grid_size}", f"{grid_size}",
                        "--phw_photonDetector.period", f"{detector_period}",
                        "--phw_photonDetector.placement", f"{placement}",
                        "--phw_photonDetector.infix", f"{infix}",
                        "--phw_photonDetector.ext", f"{backend}",
                        "--phw_photonDetector.size.x", "3",
                        "--phw_photonDetector.size.y", "3",
                        "--php_photonDetector.period", f"{detector_period}",
                        "--php_photonDetector.placement", f"{placement}",
                        "--php_photonDetector.infix", f"{infix}",
                        "--php_photonDetector.ext", f"{backend}",
                        "--php_photonDetector.size.x", "3",
                        "--php_photonDetector.size.y", "3"]
    # run simulation
    picongpu_result = subprocess.run(picongpu_command, cwd=picongpu_run_dir)
    # continue only after a successful simulation
    assert picongpu_result.returncode == 0

    # load simulation output
    if infix == "NULL":
        infix = ""
    openpmd_path = picongpu_run_dir / 'photonDetector'
    phw_series_path_str = "{}/{}{}.{}".format(str(openpmd_path),
                                              'phw_photonDetectorData', infix,
                                              backend)
    php_series_path_str = "{}/{}{}.{}".format(str(openpmd_path),
                                              'php_photonDetectorData', infix,
                                              backend)
    phw_series = api.Series(phw_series_path_str, api.Access.read_only)
    php_series = api.Series(php_series_path_str, api.Access.read_only)

    accum_policy_to_meshname = {"CountParticles": "particleCount",

                                "AddWaveParticles": "amplitude"}

    phw_iteration = phw_series.iterations[steps]
    phw_mesh = phw_iteration.meshes[accum_policy_to_meshname[accum_policy]]
    phw_mrc = phw_mesh[api.Mesh_Record_Component.SCALAR]
    phw_data = phw_mrc.load_chunk()
    php_iteration = php_series.iterations[steps]
    php_mesh = php_iteration.meshes[accum_policy_to_meshname[accum_policy]]
    php_mrc = php_mesh[api.Mesh_Record_Component.SCALAR]
    php_data = php_mrc.load_chunk()
    phw_series.flush()
    php_series.flush()

    # y,x in openPMD to x,y
    phw_data = phw_data.T
    php_data = php_data.T
    # convert to particle count
    phw_data = np.round(np.abs(phw_data)).astype(np.int32)
    php_data = np.round(np.abs(php_data)).astype(np.int32)

    model_result = np.zeros((3, 3), dtype=np.int32)
    model_result[cell_to_hit[0], cell_to_hit[1]] = 1
    # buffers in pic have (0,0) in the down left corner but arrays start
    # in the upper left corner
    model_result = model_result[::-1, :]
    phw_compare = phw_data == model_result
    php_compare = php_data == model_result

    assert phw_compare.all() and php_compare.all()
