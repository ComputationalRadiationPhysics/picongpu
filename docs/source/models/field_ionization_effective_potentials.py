import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


params = {
    'font.size': 20,
    'lines.linewidth': 3,
    'legend.fontsize': 20,
    'legend.frameon': False,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    # RTD default textwidth: 800px
    'figure.figsize': [12, 8]
}
mpl.rcParams.update(params)


def V_eff(x, Z_eff, F):
    """
    Effective radially symmetric nuclear potential under the influence of a
    homogeneous external electric field. Assumed center of potential well
    at x = 0.

    Args:
        x       : spatial coordinate
        Z_eff   : effective residual ion charge
        F       : electric field strength

    Unit:
        1 AU = 27.2 eV
    """
    E_pot = -Z_eff / np.abs(x) + F * x
    return E_pot


if __name__ == "__main__":
    """
    On execution this script produces the image
    `field_ionization_effective_potentials.svg`
    for the PIConGPU documentation.
    """

    # for C-II from Clementi et al
    Z_C_eff = 3.136
    # for C-II naively from Z-N_eb, where N_eb denotes number of
    # bound electrons after ionization
    Z_C_naive = 2.0
    # Hydrogen nuclear charge
    Z_H = 1.0

    xmin = -10
    xmax = 10

    E_AU = 27.2  # atomic unit of energy
    E_CII = 24.36 / E_AU
    E_H = 13.6 / E_AU

    # barrier suppression field strength of CII
    F_BSI_CII = E_CII**2 / (4*Z_C_eff)

    r = np.linspace(xmin, xmax, 1000)  # spatial dimension

    V_H = V_eff(r, Z_H, -F_BSI_CII)
    V_CII_naive = V_eff(r, Z_C_naive, -F_BSI_CII)
    V_CII_eff = V_eff(r, Z_C_eff, -F_BSI_CII)

    # create the figure
    plt.figure()

    # plot the effective potentials
    plt.plot(r, V_H, label="H")
    plt.plot(r, V_CII_naive, label=r"C-II with $Z = 2$")
    plt.plot(r, V_CII_eff, label=r"C-II with $Z_\mathrm{eff} = 3.136$")

    # plot the ionization energies
    plt.hlines(-E_H, xmin, xmax)
    plt.hlines(-E_CII, xmin, xmax)

    # add the legend and format the plot
    plt.title("Effective atomic potentials of Carbon-II and Hydrogen in\n"
              r"homogeneous electric field $F_\mathrm{BSI}$ (C-II)")
    plt.legend(loc="best")
    plt.text(xmin+1, -E_H+.05, r"$E_\mathrm{i}$ H")
    plt.text(xmin+1, -E_CII+.05, r"$E_\mathrm{i}$ C-II (Z_eff)")
    plt.text(xmin+1, -2.1, r"$V_\mathrm{eff} = -\frac{Z}{|x|} + Fx$")
    plt.text(xmin+1, -2.6, r"$\mathrm{\ with\ } F = F_\mathrm{BSI}$(C-II)")
    plt.xlim([-10., 10.])
    plt.ylim([-3., 1.])
    plt.ylabel(r"$V_\mathrm{eff}$ [AU = Rydberg]")
    plt.xlabel(r"$x$ [AU = Bohr radii]")

    plt.show()
