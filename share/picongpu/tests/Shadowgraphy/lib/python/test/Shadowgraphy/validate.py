"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Klaus Steiniger, Finn-Ole Carstens
License: GPLv3+
"""

import itertools
import sys

import numpy as np
import openpmd_api as io
import scipy.constants as const
import scipy.optimize as optimize


def gauss(x, amplitude, sigma, mean):
    """Gaussian function

    x: coordinate
    amplitude: amplitude
    sigma: standard deviation
    mean: mean value
    """
    exp = -((x - mean) ** 2) / (2 * sigma**2)
    return amplitude * np.exp(exp)


def test_deviation(val_simulation, val_theory, thresh, parameter_name):
    """Test function

    val_simulation: the value that should be tested
    val_theory: the value it's tested against
    thresh: the threshold to pass the test
    parameter_name: name of parameter
    """
    relative_deviation = np.abs((val_theory - val_simulation) / (val_theory))
    if relative_deviation < thresh:
        print(f"{parameter_name} passed the test with {val_simulation:.5e} compared to {val_theory:.5e}")
        return True
    else:
        print(f"{parameter_name} failed the test with {val_simulation:.5e} compared to {val_theory:.5e}")
        return False


def main(path):
    """Evaluate shadowgraphy plugin performance

    path: Path to simulation output
    """
    test_results = {}

    # Test parameters
    energy_thresh = 0.01
    w0_thresh = 0.01
    position_thresh = 0.01
    omega_thresh = 0.01
    bandwidth_thresh = 0.02

    # Simulation parameters
    wavelength = 800e-9
    w0 = 10e-6  # 2 times sigma
    tau = 10e-15  # sigma of intensity
    a0 = 1.0

    nx = 208
    ny = 208
    dx = 40e-8
    dy = 40e-8

    ########################################
    ########## Theoretical values ##########
    ########################################

    # Circular frequency
    omega = 2 * np.pi * const.c / wavelength

    # Energy of a Gaussian pulse
    electric_field = a0 * const.m_e * omega * const.c / const.e
    intensity = const.c * const.epsilon_0 * electric_field * electric_field / 2
    power = intensity * np.pi * w0 * w0 / 2
    energy_theory = power * tau * np.sqrt(np.pi * 2)

    # Focus position
    focus_x = nx * dx / 2
    focus_y = ny * dy / 2

    # Bandwidth
    tau_fwhm_intensity = 2 * np.sqrt(2 * np.log(2)) * tau
    bandwidth_fwhm_intensity = 2 * np.pi * 0.441 / tau_fwhm_intensity
    bandwidth_sigma_intensity = bandwidth_fwhm_intensity / (2 * np.sqrt(2 * np.log(2)))
    bandwidth_expected = bandwidth_sigma_intensity * np.sqrt(2)

    #########################################
    #### Calculate Shadowgram properties ####
    #########################################

    # Load data from simulation
    series = io.Series(path + "/shadowgraphy_" + "%T." + "bp5", io.Access.read_only)
    i = series.iterations[[i for i in series.iterations][0]]

    chunkdata = i.meshes["shadowgram"][io.Mesh_Record_Component.SCALAR].load_chunk()
    unit = i.meshes["shadowgram"].get_attribute("unitSI")
    series.flush()

    shadowgram = chunkdata * unit

    xspace_tmp = i.meshes["Spatial positions"]["x"].load_chunk()
    xunit = i.meshes["Spatial positions"]["x"].get_attribute("unitSI")
    series.flush()

    yspace_tmp = i.meshes["Spatial positions"]["y"].load_chunk()
    yunit = i.meshes["Spatial positions"]["y"].get_attribute("unitSI")
    series.flush()

    xspace = xspace_tmp * xunit
    yspace = yspace_tmp * yunit

    dx = xspace[0, 1] - xspace[0, 0]
    dy = yspace[1, 0] - yspace[0, 0]

    xm, ym = np.meshgrid(xspace, yspace)

    # Test energy in shadowgram
    energy_shadowgram = np.sum(shadowgram) * dx * dy
    test_results["Energy"] = test_deviation(energy_shadowgram, energy_theory, energy_thresh, "Energy")

    # Find position of maximum for lineouts
    max_position = np.unravel_index(np.argmax(shadowgram.transpose()), shadowgram.transpose().shape)

    # Test x lineout of shadowgram
    xdata = xspace[0, :]
    shadowgram_x_lineout = shadowgram[max_position[0], :]

    xbounds = [[0, dx, np.min(xdata)], [2 * np.max(shadowgram_x_lineout), np.max(xdata), np.max(xdata)]]
    poptx, pcovx = optimize.curve_fit(gauss, xdata, shadowgram_x_lineout, bounds=xbounds)

    test_results["w0_x"] = test_deviation(2 * poptx[1], w0, w0_thresh, "w0_x")
    test_results["pos_x"] = test_deviation(poptx[2], focus_x, position_thresh, "pos_x")

    # Test y lineout of shadowgram
    ydata = yspace[:, 0]
    shadowgram_y_lineout = shadowgram[:, max_position[1]]

    ybounds = [[0, dy, np.min(ydata)], [2 * np.max(shadowgram_y_lineout), np.max(ydata), np.max(ydata)]]
    popty, pcovy = optimize.curve_fit(gauss, ydata, shadowgram_y_lineout, bounds=ybounds)

    test_results["w0_y"] = test_deviation(2 * popty[1], w0, w0_thresh, "w0_y")
    test_results["pos_y"] = test_deviation(popty[2], focus_y, position_thresh, "pos_y")

    #########################################
    ###### Calculate Fourier properties #####
    #########################################
    possible_signs = ["positive", "negative"]
    possible_fields = ["Ex", "Ey", "Bx", "By"]
    for sf in itertools.product(possible_signs, possible_fields):
        series = io.Series(path + "/shadowgraphy_fourierdata_" + "%T." + "bp5", io.Access.read_only)
        i = series.iterations[[i for i in series.iterations][0]]

        chunkdata = i.meshes[f"Fourier Domain Fields - {sf[0]}"][sf[1]].load_chunk()
        unit = i.meshes[f"Fourier Domain Fields - {sf[0]}"][sf[1]].get_attribute("unitSI")
        series.flush()

        fourier_field_raw = chunkdata * unit

        xspace_tmp = i.meshes["Spatial positions"]["x"].load_chunk()
        xunit = i.meshes["Spatial positions"]["x"].get_attribute("unitSI")
        series.flush()

        yspace_tmp = i.meshes["Spatial positions"]["y"].load_chunk()
        yunit = i.meshes["Spatial positions"]["y"].get_attribute("unitSI")
        series.flush()

        xspace = xspace_tmp * xunit
        yspace = yspace_tmp * yunit

        omegaspace_tmp = i.meshes["Fourier Transform Frequencies"]["omegas"].load_chunk()
        omegaunit = i.meshes["Fourier Transform Frequencies"]["omegas"].get_attribute("unitSI")
        series.flush()

        if sf[0] == "positive":
            omegaspace = omegaspace_tmp[len(omegaspace_tmp) // 2 :] * omegaunit
        else:
            omegaspace = omegaspace_tmp[: len(omegaspace_tmp) // 2] * omegaunit

        series.close()

        dx = xspace[0, 0, 1] - xspace[0, 0, 0]
        dy = yspace[0, 1, 0] - yspace[0, 0, 0]
        domega = omegaspace[1, 0, 0] - omegaspace[0, 0, 0]

        # Only take absolute values from Fourier field
        fourier_field = np.abs(fourier_field_raw)

        max_position = np.unravel_index(np.argmax(fourier_field), fourier_field.shape)

        odata = omegaspace[:, 0, 0]
        fourier_field_lineout = fourier_field[:, max_position[1], max_position[2]]

        if sf[0] == "positive":
            fit_bounds = [[0, domega / 100, min(odata)], [2 * np.max(fourier_field_lineout), max(odata), max(odata)]]
        else:
            fit_bounds = [
                [0, np.abs(domega) / 100, min(odata)],
                [2 * np.max(fourier_field_lineout), max(np.abs(odata)), max(odata)],
            ]

        popt, pcov = optimize.curve_fit(gauss, odata, fourier_field_lineout, bounds=fit_bounds)
        # (x, amplitude, sigma, mean)
        test_results[f"[{sf[1]}-{sf[0]}] bandwidth"] = test_deviation(
            popt[1], bandwidth_expected, bandwidth_thresh, f"[{sf[1]}-{sf[0]}] bandwidth"
        )
        sign_omega = omega if (sf[0] == "positive") else -omega
        test_results[f"[{sf[1]}-{sf[0]}] omega"] = test_deviation(
            popt[2], sign_omega, omega_thresh, f"[{sf[1]}-{sf[0]}] omega"
        )

    ret_value = np.array([test_results[test] for test in test_results.keys()]).all()
    sys.exit(int(not ret_value))


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    if len(sys.argv[1:]) > 1:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    main(sys.argv[1])
