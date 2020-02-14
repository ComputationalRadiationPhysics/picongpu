"""Ionization prediction module and example.

This file is part of the PIConGPU.
Copyright 2019-2020 PIConGPU contributors
Authors: Marco Garten
License: GPLv3+
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import scipy.constants as sc


params = {
    'font.size': 20,
    'lines.linewidth': 3,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    # RTD default textwidth: 800px
    'figure.figsize': [10, 16]
}
mpl.rcParams.update(params)

# dictionary of atomic units (AU) - values in SI units
atomic_unit = {
    'electric field': sc.m_e**2 * sc.e**5 / (
        sc.hbar**4 * (4 * sc.pi * sc.epsilon_0)**3
        ),
    'intensity': sc.m_e**4 / (
        8 * sc.pi * sc.alpha * sc.hbar**9
        ) * sc.e**12 / (4 * sc.pi * sc.epsilon_0)**6,
    'energy': sc.m_e * sc.e**4 / (
        sc.hbar**2 * (4 * sc.pi * sc.epsilon_0)**2
        ),
    'time': sc.hbar**3 * (
        4 * sc.pi * sc.epsilon_0
        )**2 / sc.m_e / sc.e**4
}


def F_crit(Z, E_Ip):
    """Classical barrier suppression field strength.

    :param Z: charge state of the resulting ion
    :param E_Ip: ionization potential [unit: AU]

    :returns: critical field strength [unit: AU]
    """
    return E_Ip**2. / (4. * Z)


def n_eff(Z, E_Ip):
    """Effective principal quantum number.

    :param Z: charge state of the resulting ion
    :param E_Ip: ionization potential [unit: AU]

    :returns: effective principal quantum number
    """
    return Z / np.sqrt(2. * E_Ip)


def ADKRate(Z, E_Ip, F):
    """Ammosov-Delone-Krainov ionization rate.

    A rate model, simplified by Stirling's approximation and setting the
    magnetic quantum number m=0 like in publication [DeloneKrainov1998].

    :param Z: charge state of the resulting ion
    :param E_Ip: ionization potential [unit: AU]
    :param F: field strength [unit: AU]
    """
    nEff = np.float64(n_eff(Z, E_Ip))
    D = ((4. * Z**3.) / (F * nEff**4.))**nEff

    rate = (np.sqrt((3. * nEff**3. * F) / (np.pi * Z**3.))
            * (F * D**2.) / (8. * np.pi * Z)
            * np.exp(-(2. * Z**3.) / (3. * nEff**3. * F)))

    # set nan values due to near-zero field strengths to zero
    rate = np.nan_to_num(rate)

    return rate


def convert_a0_to_Intensity(E_in_a0, lambda_laser=800.e-9):
    """Convert electric field in a0 to intensity in SI.

    :param E_in_a0: electric field [unit: a0]
    :param lambda_laser: laser wavelength [unit: m]

    :returns: intensity [unit: W/m²]
    """
    E_in_SI = E_in_a0 \
        * sc.m_e * sc.c * 2. * sc.pi * sc.c \
        / (lambda_laser * sc.e)

    intensity = 0.5 * sc.c * sc.epsilon_0 * E_in_SI**2.

    return intensity


if __name__ == "__main__":
    """
    On execution, this file produces the image
    `field_ionization_charge_state_prediction.svg`
    for the PIConGPU documentation.
    """

    # atomic units
    AU_E_eV = atomic_unit['energy'] / sc.electron_volt  # eV
    AU_F = atomic_unit['electric field']  # V/m
    AU_I = atomic_unit['intensity']  # W/m^2
    AU_T = atomic_unit['time']  # s

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

# ============================================================================
#   Create the electric field distribution for our example.
# ============================================================================
    # laser wavelength [unit: m]
    lambda_laser = 800.e-9
    # maximum electric field in a0
    E_max_a0 = 10
    # maximum intensity
    I_max = convert_a0_to_Intensity(E_in_a0=E_max_a0)  # W/m²

    intensity_fwhm = 30.e-15  # s
    intensity_sigma = intensity_fwhm / (2. * np.sqrt(2. * np.log(2)))  # s

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
        rate_matrix[i, :] = ADKRate(cs, i_pot_AU[i], e_field_envelope_AU)


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
    electric_field_BSI = F_crit(Z, i_pot_AU)

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
    ylim_ax1 = [1e-10, 1e2]  # 1 / AU of time
    ylim_ax2 = [0, 1]
    ylim_ax3 = [1e13, 1e21]  # W/cm²
    yfill_range = np.linspace(ylim_ax3[0], ylim_ax3[1], 100)

    # customize the color scale
    color = plt.cm.rainbow(np.linspace(0, 1, Z_max + 1))[::-1]

    # creation of figure and axes
    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)

    # plotting
    ax1.axhline(y=1, ls="--", lw=1, c="k")
    # plot intensity in W/cm²
    ax3.plot(time * 1e15, intensity_envelope_SI / 1e4, c="k")

    for i in np.arange(Z_max + 1):
        if i < Z_max:
            ax1.plot(
                time * 1e15, rate_matrix[i, :],
                label="{}+ to {}+".format(
                   Z[i] - 1, Z[i]), color=color[i]
                )
            ax3.axvline(
                time_BSI[i] * 1e15,
                label="{}+ to {}+".format(
                    Z[i] - 1, Z[i]), color=color[i]
                )

        ax2.plot(
            time * 1e15, charge_dist[i, :-1],
            label="{}+".format(i), color=color[i]
            )
        ax2.fill_between(
            x=time * 1e15,
            y1=0, y2=charge_dist[i, :-1],
            color=color[i], alpha=.3
            )
        # color the regions between charge state transitions in the BSI model
        if (i > 0 and i < Z_max):
            ax3.fill_betweenx(y=yfill_range,
                              x1=time_BSI[i - 1] * 1e15,
                              x2=time_BSI[i] * 1e15,
                              color=color[i],
                              alpha=.3)
        # color the range between the earliest time and the first charge state
        # transition
        if (i == 0):
            ax3.fill_betweenx(
                y=yfill_range,
                x1=time[0] * 1e15,
                x2=time_BSI[i] * 1e15,
                color=color[i],
                alpha=.3)
        # color the range between the last charge state transition and the
        # latest time
        if (i == Z_max):
            ax3.fill_betweenx(y=yfill_range,
                              x1=time_BSI[i - 1] * 1e15,
                              x2=time[-1] * 1e15,
                              color=color[i],
                              alpha=.3)

    # set plot limits
    ax1.set_xlim(xlim)
    ax1.set_yscale("log")
    ax1.set_ylim(ylim_ax1)
    ax2.set_ylim(ylim_ax2)
    ax3.set_yscale("log")
    ax3.set_ylim(ylim_ax3)

    # labels
    ax1.set_ylabel(r"Ionization Rate $\Gamma\,\mathrm{(AU^{-1})}$")
    ax2.set_ylabel(r"Rel. Population")
    ax3.set_ylabel(r"Intensity $I\,\mathrm{(W/cm^2)}$")
    ax3.set_xlabel(r"Time $t-t_\mathrm{max}\,\mathrm{(fs)}$")

    ax1.legend(loc="lower right", fancybox=True, borderpad=1)
    ax2.legend(loc="lower right", fancybox=True, borderpad=1)
    ax3.legend(loc="lower right", fancybox=True, borderpad=1)

    fig.align_labels()
    plt.tight_layout(pad=0.4)

    plt.show()
