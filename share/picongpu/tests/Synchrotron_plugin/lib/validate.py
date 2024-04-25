import sys
import openpmd_api as io
import numpy as np
from synchrotron_lib import analytical_Propability, momentum_to_energy, quad, const


# Used when comparing two sets of histogram data with different binning
# given two set of vectors x0, y0 and x1, y1 we interpolate second set to the first set and subtract them
# interpolate y1 to x0 and subtract y0 from y1
def subtract_functions(x0, y0, x1, y1):
    minX = max(np.min(x0), np.min(x1))
    maxX = min(np.max(x0), np.max(x1))
    #
    x0 = [x for x in x0 if x >= minX and x <= maxX]
    x1 = [x for x in x1 if x >= minX and x <= maxX]
    y0 = [y for x, y in zip(x0, y0) if x >= minX and x <= maxX]
    y1 = [y for x, y in zip(x1, y1) if x >= minX and x <= maxX]

    y1 = np.interp(x0, x1, y1)  # interpolate y1 to the x0 values
    return x0, np.abs(y0 - y1) / y0


# read the parameters from the params.txt file: gamma, Heff, dt
def read_setup(folderName):
    # in params.txt we have the following parameters:
    # gamma: 1000, Heff: 1000000000000000.0, dt: 4.830524160012079e-17

    with open(folderName + "/params.txt") as f:
        lines = f.readlines()
        data = lines[0].split(",")
        gamma = data[0].split(":")[1].strip()
        Heff = data[1].split(":")[1].strip()
        dt = data[2].split(":")[1].strip()

    return float(gamma), float(Heff), float(dt)


iterNo = 4000  # iteration from which we read the data must be the same as in ../bin/validate.ci


def main(dataPath):
    print(f"Validating data in {dataPath}")
    gamma, Heff, dt = read_setup(dataPath)

    series = io.Series(dataPath + "/openPMD/simData_test_%T.bp", io.Access.read_only)
    it = series.iterations[iterNo]

    simulation_dt = it.get_attribute("dt") * it.get_attribute("unit_time")
    if abs(simulation_dt - dt) > 1e-10:  # check if the simulation dt is the same as the expected dt. 1e-10 is arbitrary
        raise SystemExit(f"Simulation dt is {simulation_dt} but expected dt is {dt}\n Test failed")

    h = it.particles["y"]  # photons
    e = it.particles["e"]  # electrons

    # photons
    h1 = h["weighting"][io.Mesh_Record_Component.SCALAR]
    w = h1[:]
    w_SI = h1.unit_SI
    series.flush()
    w *= w_SI
    h1 = h["momentum"]["y"]
    p_y = h1[:]
    p_y_SI = h1.unit_SI
    series.flush()
    p_y = np.float64(p_y)
    p_y *= p_y_SI / w
    hist_data = np.abs(p_y)
    if len(hist_data) < 5e5:
        raise SystemExit(
            f"Number of photons is {len(hist_data)} but expected number of photons to be at least 5e5\n Test failed"
        )
    hist_data = np.abs(p_y * const.c / const.elementary_charge)  # calculate the energy of photons

    # electron energy: take electron momentum and calculate energy
    h1 = e["weighting"][io.Mesh_Record_Component.SCALAR]
    e_w = h1[:]  # get the weighting
    e_w_SI = h1.unit_SI  # get the weighting unit
    series.flush()
    e_w *= e_w_SI  # convert to SI units
    h1 = e["momentum"]["y"]
    e_p_y = h1[:]  # get the momentum
    e_p_y_SI = h1.unit_SI  # get the momentum unit
    series.flush()
    e_p_y *= e_p_y_SI / e_w
    Energy_e = momentum_to_energy(e_p_y, const.m_e).max()  # convert momentum to energy

    hist_data /= Energy_e  # go to delta space

    min_exp = np.floor(np.log10(np.min(hist_data)))
    max_exp = np.ceil(np.log10(np.max(hist_data)))
    bins = np.logspace(min_exp, max_exp, 100)
    a, b = np.histogram(hist_data, bins=bins)
    normaliztion_factor = iterNo * len(e_w)

    delta = bins
    analytical_integrated = []
    for x0, x1 in zip(delta[:-1], delta[1:]):
        analytical_integrated.append(quad(lambda x: analytical_Propability(x, gamma, Heff, dt), x0, x1)[0])

    mask = a > 1000
    if mask.sum() < 5:
        print(
            f"There is less than 5 bins with 1000 photons or more. Max photons in bin = {a.max()}. Increase your statistics: more iterations, more electrons, larger dt. Higher Heff or gamma"
        )
        return 1

    # errorBar = np.sqrt(a[mask])/normaliztion_factor
    a = a / normaliztion_factor
    a = a[mask]
    b = b[1:]
    b = b[mask]
    delta = delta[1:]
    delta = delta[mask]
    analytical_integrated = np.array(analytical_integrated)[mask]

    x, y = subtract_functions(delta, analytical_integrated, b, a)
    poorness = np.sum(y) / len(y)  # average error
    poornessBound = 0.1  # 10% error. We want poorness to be less than 10%
    print(f"Poorness: {poorness}")

    retValue = int(0) if poorness <= poornessBound else int(1)

    sys.exit(retValue)


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    if len(sys.argv[1:]) > 1:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    main(sys.argv[1])
