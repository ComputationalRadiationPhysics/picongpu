"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Klaus Steiniger
License: GPLv3+
"""

import sys
import numpy as np
import openpmd_api as opmd


def main(dataPath):
    """Evaluate the absorber performance

    dataPath is supposed to point to the openPMD directory of the two
    PIConGPU simulations where the files of the test simulation and
    reference simulation are located.
    Their naming follows the pattern simData_test_%T.bp and
    simData_ref_%T.bp, respectively.

    Return zero in case of sufficient absorption at the boundary and
    one in case of too small absorption at the boundary.
    """
    series_test = opmd.Series(dataPath + "/simData_test_%T.bp", opmd.Access.read_only)
    series_ref = opmd.Series(dataPath + "/simData_ref_%T.bp", opmd.Access.read_only)

    """Read simulation parameters"""
    # shape of a data set
    Nt = len(series_test.iterations)
    Nz, Ny, Nx = series_test.iterations[0].meshes["B"]["x"].shape
    Nz_ref, Ny_ref, Nx_ref = series_ref.iterations[0].meshes["B"]["x"].shape

    """ Define measurement position
        Compared to sec. 7.11.1 'Current Source Radiating in an Unbounded
        Two-Dimensional Region' of the Taflove book, point A corresponds
        to an offset only along x, while point B corresponds to an offset
        along x and y.
    """
    # Define the offset of the measurement point from the center of the
    # conductor in cells.
    # The conductor is located in the center of simulation domain.
    # The offset is choosen such that the measurement point is located 2 cells
    # in front of the PML in the small volume.
    offset_cells = int(18)
    x_offset = -offset_cells
    y_offset = -offset_cells
    z_offset = int(0)

    # Calculate absolute position of measurement point in the small volume.
    x_meas = Nx // 2 + x_offset
    y_meas = Ny // 2 + y_offset
    z_meas = Nz // 2 + z_offset

    # Calculate absolute position of measurement point in the reference sim.
    x_meas_ref = Nx_ref // 2 + x_offset
    y_meas_ref = Ny_ref // 2 + y_offset
    z_meas_ref = Nz_ref // 2 + z_offset

    """Load data"""
    times = np.zeros(Nt)

    # create data containers.
    # Explicitly provide data type in order to circumvent
    # "RuntimeError: [Record_Component::load_chunk()]
    #  Requires contiguous slab of memory."
    # of openPMD
    Ex = np.zeros(Nt, dtype=np.float32)
    Ey = np.zeros(Nt, dtype=np.float32)
    Ez = np.zeros(Nt, dtype=np.float32)

    Ex_ref = np.zeros(Nt, dtype=np.float32)
    Ey_ref = np.zeros(Nt, dtype=np.float32)
    Ez_ref = np.zeros(Nt, dtype=np.float32)

    for i, it in enumerate(series_test.iterations):
        times[i] = it

        Efield = series_test.iterations[it].meshes["E"]

        unit_fieldE = Efield["x"].unit_SI

        ##################
        # Load test data #
        ##################

        Efield["x"].load_chunk(Ex[i : i + 1], offset=(z_meas, y_meas, x_meas), extent=(1, 1, 1))
        Efield["y"].load_chunk(Ey[i : i + 1], offset=[z_meas, y_meas, x_meas], extent=[1, 1, 1])
        Efield["z"].load_chunk(Ez[i : i + 1], offset=[z_meas, y_meas, x_meas], extent=[1, 1, 1])

        # spare the series.flush() since the iteration is closed immediately

        series_test.iterations[it].close()

        Ex[i] *= unit_fieldE
        Ey[i] *= unit_fieldE
        Ez[i] *= unit_fieldE

        #######################
        # Load reference data #
        #######################
        Efield_ref = series_ref.iterations[it].meshes["E"]

        Efield_ref["x"].load_chunk(
            Ex_ref[i : i + 1],
            offset=[z_meas_ref, y_meas_ref, x_meas_ref],
            extent=[1, 1, 1],
        )
        Efield_ref["y"].load_chunk(
            Ey_ref[i : i + 1],
            offset=[z_meas_ref, y_meas_ref, x_meas_ref],
            extent=[1, 1, 1],
        )
        Efield_ref["z"].load_chunk(
            Ez_ref[i : i + 1],
            offset=[z_meas_ref, y_meas_ref, x_meas_ref],
            extent=[1, 1, 1],
        )

        # spare the series_ref.flush()
        # since the iteration is closed immediately

        series_ref.iterations[it].close()

        Ex_ref[i] *= unit_fieldE
        Ey_ref[i] *= unit_fieldE
        Ez_ref[i] *= unit_fieldE

    """ Calculate the quantity defining the
        quality/performance of the absorber.
    """
    component, component_ref = (
        np.sqrt(Ex**2 + Ey**2 + Ez**2),
        np.sqrt(Ex_ref**2 + Ey_ref**2 + Ez_ref**2),
    )

    quality = np.abs(component - component_ref) / np.abs(component_ref).max()

    """ Evaluate success or failure of the test
        As of 2023-09-12 we have seen values of the quality <=4.e-5
        for point B, which has an offset=(x,y,z)=(-18, -18, 0),
        and values <=6.e-5 for point A,
        which has an offset=(x,y,z)=(-18,0,0) from the conductor.
        This is in the same range as the values shown in Taflove Fig. 7.3(b)
        for a 10-cell CPML.
        We therefore trust our current implementation and set the following
        bound on the relative deviation between reference and test simulation.
    """
    qualityBound = 1.0e-4

    retValue = int(0) if quality.max() <= qualityBound else int(1)

    sys.exit(retValue)


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    if len(sys.argv[1:]) > 1:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    main(sys.argv[1])
