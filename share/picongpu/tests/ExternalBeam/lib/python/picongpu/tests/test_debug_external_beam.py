import os
import pytest
import subprocess
from pathlib import Path
import numpy as np

from verify_results import verify_results


@pytest.mark.parametrize("side_str,offset,yaw,pitch",
                         [
                             # test simplest case for all sides
                             ('x', (0, 0), 0, 0),  # 0
                             ('xr', (0, 0), 0, 0),  # 1
                             ('y', (0, 0), 0, 0),  # 2
                             ('yr', (0, 0), 0, 0),  # 3
                             ('z', (0, 0), 0, 0),  # 4
                             ('zr', (0, 0), 0, 0),  # 5
                             # test offset only for one side
                             ('x', (5, 2), 0, 0),  # 6
                             # test yaw only for one side
                             ('x', (0, 0), 30.0, 0),  # 7
                             # test pitch only for one side
                             ('x', (0, 0), 0, 40.0),  # 8
                             # test yaw and pitch for one side
                             ('x', (0, 0), 30.0, 40.0),  # 9
                             # test the most general case
                             # (offset + both angles)
                             # for all sides
                             ('x', (5, 2), 30.0, 40.0),  # 10
                             ('xr', (5, 2), 30.0, 40.0),  # 11
                             ('y', (5, 2), 30.0, 40.0),  # 12
                             ('yr', (5, 2), 30.0, 40.0),  # 13
                             ('z', (5, 2), 30.0, 40.0),  # 14
                             ('zr', (5, 2), 30.0, 40.0)  # 15
                         ])
def test_detect_single_particle(tmp_path, side_str,
                                offset, yaw,
                                pitch):
    # use pic create to create a tmp example dir
    example_root = tmp_path / "example"
    example_src = Path(os.environ.get(
        "PICSRC")) / 'share' / 'picongpu' / 'tests' / 'ExternalBeam'

    create_result = subprocess.run(
        ["pic-create", f"{str(example_src.resolve())}",
         f"{str(example_root.resolve())}"], check=True)
    assert create_result.returncode == 0

    # some simulation params
    steps = 1
    grid_size_x, grid_size_y, grid_size_z = 9, 9, 9
    cell_size = 0.1772e-6
    debug_period = "1"
    sigma = 5

    side_map = {'x': 'XSide', 'xr': 'XRSide', 'y': 'YSide', 'yr': 'YRSide',
                'z': 'ZSide', 'zr': 'ZRSide'}
    # prepare  cmake flags

    overwrite_list = f"-DPARAM_YAW_ANGLE={yaw:e}_X;" \
                     f"-DPARAM_PITCH_ANGLE={pitch:e}_X;" \
                     f"-DPARAM_BEAM_OFFSET_X={offset[0] * cell_size:e}_X;" \
                     f"-DPARAM_BEAM_OFFSET_Y={offset[1] * cell_size:e}_X;" \
                     f"-DPARAM_SIGMA_X_SI={sigma * cell_size:e}_X;" \
                     f"-DPARAM_SIGMA_Y_SI={sigma * cell_size:e}_X;" \
                     f"-DPARAM_BEAM_PROFILE=GaussianProfile;" \
                     f"-DPARAM_BEAM_SIDE={side_map[side_str]};" \
                     f"-DPARAM_BEAM_SHAPE=ConstShape;"
    cmake_flags = f"-DPARAM_OVERWRITES:LIST='{overwrite_list}'"

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
                        "-g", f"{grid_size_x}", f"{grid_size_y}",
                        f"{grid_size_z}",
                        "--periodic", "1", "1", "1",
                        "--debugExternalBeam.period", f"{debug_period}"]
    # run simulation
    picongpu_result = subprocess.run(picongpu_command, cwd=picongpu_run_dir)
    # continue only after a successful simulation
    assert picongpu_result.returncode == 0

    checks = verify_results(picongpu_run_dir, sigma * cell_size, side_str,
                            (grid_size_x, grid_size_y, grid_size_z),
                            yaw, pitch, offset)
    print(checks)
    assert np.all(checks[0])
