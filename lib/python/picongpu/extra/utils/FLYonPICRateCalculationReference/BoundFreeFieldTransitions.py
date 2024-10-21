#!/usr/bin/env python
"""
atomicPhysics(FLYonPIC) reference rate calculation
This file is part of the PIConGPU.
Copyright 2023-2024 PIConGPU contributors
Authors: Brian Marre, Marco Garten
License: GPLv3+
"""

import numpy as np
import scipy.constants as sc
import mpmath as mp

""" @file reference implementation of the rate calculation for bound-free field based transitions """

# set decimal precision of mpmath to be used in calculation
mp.mp.dps = 200

# dictionary of atomic units (AU) - values in SI units
atomic_unit = {
    "electric field": sc.m_e**2 * sc.e**5 / (sc.hbar**4 * (4 * sc.pi * sc.epsilon_0) ** 3),
    "intensity": sc.m_e**4 / (8 * sc.pi * sc.alpha * sc.hbar**9) * sc.e**12 / (4 * sc.pi * sc.epsilon_0) ** 6,
    "energy": sc.m_e * sc.e**4 / (sc.hbar**2 * (4 * sc.pi * sc.epsilon_0) ** 2),
    "time": sc.hbar**3 * (4 * sc.pi * sc.epsilon_0) ** 2 / sc.m_e / sc.e**4,
}


class BoundFreeFieldTransitions:
    @staticmethod
    def F_crit_BSI(Z, E_Ip):
        """Classical barrier suppression field strength.

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]

        :returns: critical field strength [unit: AU]
        """
        return E_Ip**2 / (4.0 * Z)

    @staticmethod
    def n_eff_mpmath(Z, E_Ip):
        """Effective principal quantum number as defined in the ADK rate model.

        @detail version using arbitrary precision library mpmath for calculation

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]

        :returns: effective principal quantum number
        """
        return Z / mp.sqrt(mp.mpf(2) * E_Ip)

    @staticmethod
    def n_eff_numpy(Z, E_Ip):
        """Effective principal quantum number as defined in the ADK rate model.

        @detail version using fixed numpy functions for calculation, same precision as PIConGPU implementation

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]

        :returns: effective principal quantum number
        """
        return np.float32(Z / np.sqrt(2.0 * E_Ip))

    @staticmethod
    def ADKRate_numpy(Z, E_Ip, F):
        """Ammosov-Delone-Krainov ionization rate.

        A rate model, simplified by Stirling's approximation and setting the
        magnetic quantum number m=0 like in publication [DeloneKrainov1998].

        @detail version using fixed numpy functions for calculation, same precision as PIConGPU implementation

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]
        :param F: field strength [unit: AU]

        :returns: ionization rate [unit: 1/AU(time)]
        """

        nEff = BoundFreeFieldTransitions.n_eff_numpy(Z, E_Ip)

        dBase = np.float64(4.0 * np.exp(1.0) * Z**3.0 / (F * nEff**4.0))
        D = dBase ** np.float64(nEff)

        print("\t\t nEff: {:.4e}".format(nEff))
        print("\t\t dBase: {:.4e}".format(dBase))
        print("\t\t D: {:.4e}".format(D))

        gamowFactor = np.exp(-(2.0 * Z**3.0) / (3.0 * nEff**3.0 * F))
        print("\t\t gamowFactor: {:.4e}".format(gamowFactor))

        rate = (
            F * np.float32(D**2.0 * gamowFactor) * np.sqrt((3.0 * nEff**3.0 * F) / (np.pi * Z**3.0)) / (8.0 * np.pi * Z)
        )
        return rate

    @staticmethod
    def ADKRate_mpmath(Z, E_Ip, F):
        """Ammosov-Delone-Krainov ionization rate.

        A rate model, simplified by Stirling's approximation and setting the
        magnetic quantum number m=0 like in publication [DeloneKrainov1998].

        @detail version using arbitrary precision library mpmath for calculation

        :param Z: charge state of the resulting ion
        :param E_Ip: ionization potential [unit: AU]
        :param F: field strength [unit: AU]

        :returns: ionization rate [unit: 1/AU(time)]
        """

        nEff = BoundFreeFieldTransitions.n_eff_mpmath(Z, E_Ip)

        dBase = (mp.mpf(4) * mp.e * Z ** mp.mpf(3)) / (F * nEff ** mp.mpf(4))
        D = dBase**nEff

        rate = (
            (F * D ** mp.mpf(2))
            / (mp.mpf(8) * mp.pi * Z)
            * mp.exp(-(mp.mpf(2) * Z ** mp.mpf(3)) / (mp.mpf(3) * nEff ** mp.mpf(3) * F))
            * mp.sqrt((mp.mpf(3) * nEff ** mp.mpf(3) * F) / (mp.pi * Z ** mp.mpf(3)))
        )

        return rate
