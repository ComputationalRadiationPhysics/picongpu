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


def ADK_rate_simple(Z, E_i, F):
    """
    ADK tunneling rate from [DeloneKrainov1998] that has been simplified
    for s-states.

    Args:
        Z   : residual ion charge
        E_i : ionization energy of the state
        F   : electric field strength
    """
    n_eff = Z / ((2.*E_i)**(1./2.))     # effective principal quantum number
    D = ((4.*np.exp(1.)*Z**3.) / (F*n_eff**4.))**n_eff         # some factor

    # laser is circularly polarized
    # pol_fac = 1.
    # laser is lin. polarized
    pol_fac = ((3*n_eff**3*F) / (np.pi*Z**3))**(1.0/2)
    I_rate = pol_fac * (F*D**2.) / (8.*np.pi*Z) * \
        np.exp(-(2.*Z**3.) / (3.*n_eff**3.*F))
    return I_rate


if __name__ == "__main__":
    """
    On execution this file produces the image
    `field_ionization_comparison_c_ii_ionization.svg`
    for the PIConGPU documentation.
    """

    # Atomic units
    E_AU = 27.2  # eV
    F_AU = 5.1422e11  # V/m
    I_AU = 3.5095e16  # W/cm^2
    T_AU = 150.e-18 / (2. * np.pi)

    # Hydrogen
    Z_H = 1.  # proton number
    E_H = .5  # ionization energy (AU)

    # Carbon
    Z_C = 6.
    E_C_SI = np.array([11.26, 24.36, 47.89, 64.49, 392.09, 490.00])
    E_C = E_C_SI/E_AU

    ymin = 1e6
    ymax = 1e20

    # electric field strengths in AU
    fields = 10**np.linspace(-5, 3, 1000)

    H = ADK_rate_simple(Z_H, E_H, fields)/T_AU

    Csimple = ADK_rate_simple(2,  E_C[1], fields) / T_AU
    # with effective Z from shielding constants
    Ceff = ADK_rate_simple(3.136, E_C[1], fields) / T_AU

    plt.figure()

    p_H = plt.plot(fields, H, label="ADK H")

    p_Csimple = plt.plot(fields, Csimple,
                         label=r"ADK C  $ \quad Z = 2\mathrm{+}$")
    p_Ceff = plt.plot(fields, Ceff,
                      label=r"ADK C (eff) $ \quad Z = 3.136 = Z_\mathrm{eff}$")
    plt.vlines(E_H**2./(4*1), ymin, ymax,
               colors="{}".format(p_H[0].get_color()),
               label=r"$F_\mathrm{BSI}$ H", linestyles="--")

    plt.vlines(E_C[1]**2. / (4*2), ymin, ymax,
               colors="{}".format(p_Csimple[0].get_color()),
               label=r"$F_\mathrm{BSI}$ C", linestyles="--")
    plt.vlines(E_C[1]**2. / (4*3.136), ymin, ymax,
               colors="{}".format(p_Ceff[0].get_color()),
               label=r"$F_\mathrm{BSI}$ C (eff)", linestyles="--")

    plt.title("Comparison of ADK ionization rates for\nCarbon-II and Hydrogen")
    plt.ylim([ymin, ymax])
    plt.xlim([1e-2, 1e1])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(r"ionization rate $\Gamma$ [s$^{-1}$]")
    plt.xlabel(r"field strength $F$ [AU = 5.1422$\cdot 10^{11}$ V/m]")
    plt.legend(loc="best")

    plt.show()
