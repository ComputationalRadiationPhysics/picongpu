"""Field ionization models implemented in PIConGPU.

This file is part of the PIConGPU.
Copyright 2019-2020 PIConGPU contributors
Authors: Marco Garten
License: GPLv3+
"""


import numpy as np
import scipy.constants as sc


class FieldIonization:
    """Field ionization class, containing methods and units.

    A field ionization class that contains functions to calculate ionization
    rates, threshold fields and atomic units.
    """

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

    def F_crit_BSI(self,
                   Z, E_Ip):
        """Classical barrier suppression field strength.

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]

        :returns: critical field strength [unit: AU]
        """
        return E_Ip**2. / (4. * Z)

    def F_crit_BSIStarkShifted(self,
                               E_Ip):
        """Barrier suppression field strength according to \
        Bauer 2010 - High Power Laser Matter Interaction, p. 276, Eq. (7.45).

        :param E_Ip: ionization potential [unit: AU]

        :returns: critical field strength [unit: AU]
        """
        return (np.sqrt(2.) - 1.) * E_Ip**(3./2.)

    def n_eff(self,
              Z, E_Ip):
        """Effective principal quantum number.

        Belongs to the ADK rate model.

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]

        :returns: effective principal quantum number
        """
        return Z / np.sqrt(2. * E_Ip)

    def ADKRate(self,
                Z, E_Ip, F, polarization="linear"):
        """Ammosov-Delone-Krainov ionization rate.

        A rate model, simplified by Stirling's approximation and setting the
        magnetic quantum number m=0 like in publication [DeloneKrainov1998].

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]
        :param F: field strength [unit: AU]
        :param polarization: laser polarization
                             ['linear' (default), 'circular']

        :returns: ionization rate [unit: 1/AU(time)]
        """
        pol = polarization
        if pol not in ["linear", "circular"]:
            raise NotImplementedError(
                "Cannot interpret polarization='{}'.\n".format(pol) +
                "So far, the only implemented options are: " +
                "['linear', 'circular']"
                )

        nEff = np.float64(self.n_eff(Z, E_Ip))
        D = ((4. * Z**3.) / (F * nEff**4.))**nEff

        rate = (F * D**2.) / (8. * np.pi * Z) \
            * np.exp(-(2. * Z**3.) / (3. * nEff**3. * F))

        if pol == 'linear':
            rate = rate \
                * np.sqrt((3. * nEff**3. * F) / (np.pi * Z**3.))

        # set nan values due to near-zero field strengths to zero
        rate = np.nan_to_num(rate)

        return rate

    def KeldyshRate(self,
                    E_Ip, F):
        """Keldysh model ionization rate.

        :param E_Ip: ionization potential [unit: AU]
        :param F: field strength [unit: AU]

        :returns: ionization rate [unit: 1/AU(time)]
        """
        # characteristic exponential function argument
        charExpArg = np.sqrt((2.*E_Ip)**3) / F

        # ionization rate
        rate = np.sqrt(6.*np.pi) / 2**(5./4.) \
            * E_Ip * np.sqrt(1./charExpArg) \
            * np.exp(-2./3. * charExpArg)

        return rate

    @staticmethod
    def convert_a0_to_Intensity(E_in_a0, lambda_laser=800.e-9):
        """Convert electric field in a0 to intensity in SI.

        :param E_in_a0: electric field [unit: a0]
        :param lambda_laser: laser wavelength [unit: m]

        :returns: intensity [unit: W/m^2]
        """
        E_in_SI = E_in_a0 \
            * sc.m_e * sc.c * 2. * sc.pi * sc.c \
            / (lambda_laser * sc.e)

        intensity = 0.5 * sc.c * sc.epsilon_0 * E_in_SI**2.

        return intensity
