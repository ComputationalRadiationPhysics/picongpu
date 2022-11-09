"""
This file is part of the PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""
import os
import subprocess
import json
import argparse
from pathlib import Path


def compile_setup(example_src, example_root, side_str, offset):
    """Create example and compile it
    :param example_src: pathlib Path object referring to example source dir
    :param example_root: pathlib Path object referring to directory where the
        example will be created and compiled
    :param side_str: string defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
    """
    create_result = subprocess.run(["pic-create",
                                    f"{str(example_src.resolve())}",
                                    f"{str(example_root.resolve())}"],
                                   check=True)
    assert create_result.returncode == 0

    # some simulation params

    cell_size = 3e-6  # in meters
    sigma = (10, 7)  # in cells x,y beam system coordinates
    start_time = 0  # time_steps
    end_time = 20  # time_steps

    side_map = {'x': 'XSide', 'xr': 'XRSide', 'y': 'YSide', 'yr': 'YRSide',
                'z': 'ZSide', 'zr': 'ZRSide'}
    # prepare  cmake flags

    overwrite_list = f"-DPARAM_BEAM_OFFSET_X={offset[0] * cell_size:e}_X;" \
                     f"-DPARAM_BEAM_OFFSET_Y={offset[1] * cell_size:e}_X;" \
                     f"-DPARAM_SIGMA_X_SI={sigma[0] * cell_size:e}_X;" \
                     f"-DPARAM_SIGMA_Y_SI={sigma[1] * cell_size:e}_X;" \
                     f"-DPARAM_BEAM_PROFILE=GaussianProfile;" \
                     f"-DPARAM_BEAM_SIDE={side_map[side_str]};" \
                     f"-DPARAM_BEAM_SHAPE=ConstShape;" \
                     f"-DPARAM_BEAM_START_T={start_time};" \
                     f"-DPARAM_BEAM_END_T={end_time};"
    cmake_flags = f"-DPARAM_OVERWRITES:LIST='{overwrite_list}'"

    # compile
    compile_command = ["pic-build", "-c", f"\"{cmake_flags}\""]
    compile_result = subprocess.run(compile_command, cwd=example_root)
    # continue only after a successful compilation
    assert compile_result.returncode == 0


def run_setup(picongpu_run_dir, picongpu_exec):
    """ Run the compiled example

    :param picongpu_run_dir: run directory, pathlib Path object
    :param picongpu_exec: picongpu executable, pathlib Path object
    """
    steps = 93
    grid_size_x, grid_size_y, grid_size_z = 24, 24, 24
    debug_period = "1"

    picongpu_command = ["mpiexec", "-n", "1", f"{picongpu_exec.resolve()}",
                        "-d", "1", "1", "1", "-s", f"{steps}", "-g",
                        f"{grid_size_x}", f"{grid_size_y}", f"{grid_size_z}",
                        "--openPMD.period", f"{debug_period}",
                        "--openPMD.file", "simData", "--openPMD.source",
                        "fields_all", "--openPMD.period", "5:5",
                        "--openPMD.file", "particles", "--openPMD.source",
                        "species_all"]
    # run simulation
    picongpu_result = subprocess.run(picongpu_command, cwd=picongpu_run_dir)
    # continue only after a successful simulation
    assert picongpu_result.returncode == 0


def test_external_beam(top_path,
                       side_str,
                       offset,
                       compile_only=False,
                       run_only=False):
    """ Compile and run external beam test

    :param top_path: test directory, pathlib Path object
    :param side_str: string defining from which side the beam enters the
        simulation. Need to be one of the following: "x", "xr", "y", "yr",
        "z", "zr". r means the beam propagates against the given axis.
    :param offset: offset in cells from the default beam center position
        in the transversal beam plane. (x,y) in beam coordinate system.
    :param compile_only: if True only compile the setup
    :param run_only:  if True only run the setup, it needs to be already
        compiled in top_path/example.
    """
    # use pic create to create a tmp example dir
    example_root = top_path / "example"
    example_src = Path(os.environ.get("PICSRC")) / 'share' / 'picongpu'
    example_src = example_src / 'tests' / 'ExternalBeam'

    if not run_only:
        compile_setup(example_src, example_root, side_str, offset)
    if not compile_only:
        picongpu_run_dir = top_path / "simOutput"
        picongpu_run_dir.mkdir()
        picongpu_exec = example_root / 'bin' / 'picongpu'
        run_setup(picongpu_run_dir, picongpu_exec)

        # write test parameters in a json file
        parameters = {"side_str": side_str, "offset": offset}
        with open(picongpu_run_dir / "test_setup.json", "w") as f:
            json.dump(parameters, f)


if __name__ == '__main__':
    # define test cases: (side_str,offset)
    setups = [  # test simplest case for all sides
        ('x', (0, 0)),  # 0
        ('xr', (0, 0)),  # 1
        ('y', (0, 0)),  # 2
        ('yr', (0, 0)),  # 3
        ('z', (0, 0)),  # 4
        ('zr', (0, 0)),  # 5
        # test offset only for one side
        ('x', (5, 2),)]  # 6

    parser = argparse.ArgumentParser(description="Run an ExternalBeam test")
    parser.add_argument('--dir',
                        nargs='?',
                        help="directory used for compiling and running"
                             " the test. Default is the current directory",
                        default=Path.cwd())
    parser.add_argument('-t',
                        type=int,
                        choices=range(len(setups)),
                        help="setup to run. Possible setups "
                             f"(side_str, offset): {setups}")
    parser.add_argument('--compile-only', default=False, action='store_true')
    parser.add_argument('--run-only', default=False, action='store_true')
    args = parser.parse_args()
    top_path = Path(args.dir)
    test_external_beam(top_path, *setups[args.t],
                       compile_only=args.compile_only, run_only=args.run_only)
