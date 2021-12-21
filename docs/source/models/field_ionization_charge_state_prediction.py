"""Ionization prediction module and example.

This file is part of the PIConGPU.
Copyright 2019-2021 PIConGPU contributors
Authors: Marco Garten
License: GPLv3+
"""

import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from importlib import import_module

picongpu_package_path = os.path.abspath('../../../lib/python/')

if picongpu_package_path not in sys.path:
    sys.path.insert(0, picongpu_package_path)


# import my own modules without having to write a 'noqa' comment because PEP8
# requires all imports to be at the top of the file
FI_module = import_module(name='.utils.field_ionization', package='picongpu')


params = {
    'font.size': 20,
    'lines.linewidth': 3,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    # RTD default textwidth: 800px
    'figure.figsize': [10, 16],
    'legend.title_fontsize': 10
}
mpl.rcParams.update(params)


def time_AU_to_SI(t_AU):
    """Convert time from AU to SI.

    :param t_AU: time in atomic units

    :returns: time in SI units
    """
    return t_AU * AU_T


def time_SI_to_AU(t_SI):
    """Convert time from SI to AU.

    :param t_SI: time in SI units

    :returns: time in atomic units
    """
    return t_SI / AU_T


if __name__ == "__main__":
    """
    On execution, this file produces the image
    `field_ionization_charge_state_prediction.svg`
    for the PIConGPU documentation.
    """
    # instantiate FieldIonization classobject
    FI = FI_module.FieldIonization()
    AU = FI.atomic_unit

    # atomic units
    AU_E_eV = AU['energy'] / sc.electron_volt  # eV
    AU_F = AU['electric field']  # V/m
    AU_I = AU['intensity']  # W/m^2
    AU_T = AU['time']  # s

    # proton number: Neon
    Z_max = 10
    # array of ionized charge states: Neon
    Z = np.arange(1, Z_max + 1, dtype=int)
    # array of ionization energies in eV for Neon
    # taken from https://physics.nist.gov/cgi-bin/ASD/ie.pl
    ionization_energies_eV = np.array([
        21.5645,
        40.9630,
        63.4233,
        97.1900,
        126.247,
        157.934,
        207.271,
        239.0969,
        1195.8078,
        1362.1991
    ])

    # convert ionization energies to atomic units
    i_pot_AU = ionization_energies_eV / AU_E_eV

    # for population conversion
    percent = 1e-2
    # femtosecond: for time conversion
    fs = 1e-15
    # cm^2 in m^2
    cm2 = 1e-4

# ============================================================================
#   Create the electric field distribution for our example.
# ============================================================================
    # laser wavelength [unit: m]
    lambda_laser = 800.e-9
    # maximum electric field in a0
    E_max_a0 = 10
    # maximum intensity
    I_max = FI.convert_a0_to_Intensity(E_in_a0=E_max_a0)  # W/m^2

    intensity_fwhm = 30.e-15  # s
    intensity_sigma = intensity_fwhm / (2. * np.sqrt(2. * np.log(2)))  # s

    # the sampling resolution here determines how smooth the transitions
    # in the Markov chain are
    t_res = 10000
    time = np.linspace(-200e-15, 200e-15, t_res)  # s
    intensity_envelope_SI = I_max * np.exp(- .5 * time**2 / intensity_sigma**2)

    intensity_envelope_AU = intensity_envelope_SI / AU_I
    e_field_envelope_AU = np.sqrt(intensity_envelope_AU)

# =============================================================================
#   Calculate the ADK rates for the electric field envelope and each charge
#   state.
# =============================================================================
    rate_matrix = np.zeros([len(i_pot_AU), t_res])

    for i, cs in enumerate(Z):
        rate_matrix[i, :] = FI.ADKRate(cs, i_pot_AU[i], e_field_envelope_AU)


# =============================================================================
#   Markovian approach for calculating the transition matrices of the problem.
# =============================================================================

    # transition matrix
    trans_mat_base = np.diag(np.ones([Z_max + 1]))
    trans_mat_before = trans_mat_base
    # preparation of the transition matrix: Markov absorbing state CS = 10
    trans_mat_base[Z_max, Z_max] = 1

    # prepare initial state
    initState = np.zeros([Z_max + 1])
    # all atoms are initially unionized
    initState[0] = 1

    # prepare expected charge distribution array
    charge_dist = np.zeros([Z_max + 1, t_res + 1])
    # manipulate last entry for loop
    charge_dist[:, -1] = initState

    # time step of the Markov process
    dt = (time[-1] - time[0]) / AU_T / t_res

    # loop over steps
    for i in np.arange(t_res):
        # calculate the transition matrix of this step
        trans_mat = trans_mat_base
        for k, cs in enumerate(Z):
            # probability to stay bound
            trans_mat[k, k] = np.exp(-rate_matrix[k, i] * dt)
            # probability to ionize
            trans_mat[k + 1, k] = 1. - np.exp(-rate_matrix[k, i] * dt)

        # Markov step
        charge_dist[:, i] = np.dot(charge_dist[:, i - 1], trans_mat.T)


# =============================================================================
#   Barrier suppression field strength calculation
# =============================================================================
    electric_field_BSI = FI.F_crit_BSI(Z, i_pot_AU)

    # find times when BSI fields are exceeded
    time_BSI = np.zeros([Z_max])
    for i, cs in enumerate(Z):
        idx = np.where(e_field_envelope_AU > electric_field_BSI[i])[0][0]
        time_BSI[i] = time[idx - 1]


# =============================================================================
#   Plotting the ionization rates, the electric field and the charge state
#   population over time.
# =============================================================================
    xlim = [-75, -0]
    ylim_ax_rate = [1e-10, 1e3]  # 1 / AU of time
    ylim_ax_pop = [0, 100]
    ylim_ax_bsi = [0, 100]
    ylim_ax_bsi_twin = [1e13, 1e21]  # W/cm^2
    yfill_range = np.linspace(ylim_ax_bsi[0], ylim_ax_bsi[1], 100)

    # customize the color scale
    color = plt.cm.rainbow(np.linspace(0, 1, Z_max + 1))[::-1]

    # creation of figure and axes
    fig = plt.figure()

    ax_rate = fig.add_subplot(312)
    ax_pop = fig.add_subplot(311, sharex=ax_rate)
    ax_bsi = fig.add_subplot(313, sharex=ax_rate)

    for i in np.arange(Z_max + 1):
        if i < Z_max:
            ax_rate.plot(
                time / fs, rate_matrix[i, :],
                label="{}+ to {}+".format(
                    Z[i] - 1, Z[i]), color=color[i]
            )
            ax_bsi.axvline(
                time_BSI[i] / fs,
                label="{}+ to {}+".format(
                    Z[i] - 1, Z[i]), color=color[i]
            )

        ax_pop.plot(
            time / fs, charge_dist[i, :-1] / percent,
            label="{}+".format(i), color=color[i]
        )
        ax_pop.fill_between(
            x=time / fs,
            y1=0, y2=charge_dist[i, :-1] / percent,
            color=color[i], alpha=.3
        )
        # color the regions between charge state transitions in the BSI model
        if (i > 0 and i < Z_max):
            ax_bsi.fill_betweenx(
                y=yfill_range,
                x1=time_BSI[i - 1] / fs,
                x2=time_BSI[i] / fs,
                color=color[i],
                alpha=.3
            )
        # color the range between the earliest time and the first charge state
        # transition
        if (i == 0):
            ax_bsi.fill_betweenx(
                y=yfill_range,
                x1=time[0] / fs,
                x2=time_BSI[i] / fs,
                color=color[i],
                alpha=.3
            )
        # color the range between the last charge state transition and the
        # latest time
        if (i == Z_max):
            ax_bsi.fill_betweenx(
                y=yfill_range,
                x1=time_BSI[i - 1] / fs,
                x2=time[-1] / fs,
                color=color[i],
                alpha=.3
            )

    ax_bsi_twin = ax_bsi.twinx()
    # plot intensity in W/cm^2
    ax_bsi_twin.plot(time / fs, intensity_envelope_SI / 1e4, c="k")

    locs = np.array([0, 100])
    labels = ["none", "all"]
    ax_bsi.set_yticks(locs)
    ax_bsi.set_yticklabels(labels)

    # secondy y-axis for rate plot
    secaxy_rate = ax_rate.secondary_yaxis(
        'right',
        functions=(time_SI_to_AU, time_AU_to_SI)
    )
    secaxy_rate.set_ylabel(r'ionizations per second')

    # set plot limits
    ax_rate.set_xlim(xlim)
    ax_rate.set_yscale("log")
    ax_rate.set_ylim(ylim_ax_rate)
    ax_pop.set_ylim(ylim_ax_pop)
    ax_bsi.set_ylim(ylim_ax_bsi)
    ax_bsi_twin.set_yscale("log")
    ax_bsi_twin.set_ylim(ylim_ax_bsi_twin)

    # note string for ADK rate plot note
    note_string = "Note: ADK rates were calculated from the " \
        + "intensity envelope below"
    # note in ADK rate plot
    ax_rate.text(
        x=0.05,
        y=0.9,
        s=note_string,
        fontsize=12,
        transform=ax_rate.transAxes
    )

    # labels
    ax_rate.set_ylabel(r"ADK: Ionization Rate $\Gamma\,\mathrm{(AU^{-1})}$")
    ax_pop.set_ylabel(r"ADK: Relative Population (%)")
    ax_bsi_twin.set_ylabel(r"Laser Intensity $I\,\mathrm{(W/cm^2)}$")
    ax_bsi.set_xlabel(r"Time $t-t_\mathrm{max}\,\mathrm{(fs)}$")
    ax_bsi.set_ylabel(r"BSI: Relative Population (%)")

    ax_rate.legend(loc="lower right", fancybox=True, borderpad=1,
                   title="charge state"+"\n"+"transition")
    ax_pop.legend(loc="lower right", fancybox=True, borderpad=1,
                  title="charge state")
    ax_bsi.legend(loc="lower right", fancybox=True, borderpad=1,
                  title="charge state"+"\n"+"transition")

    fig.align_labels()
    plt.tight_layout(pad=0.4)

    plt.show()
