#!/usr/bin/env python

# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
atomicPhysics(FLYonPIC) reference rate calculation
This file is part of the PIConGPU.
Copyright 2023-2024 PIConGPU contributors
Authors: Brian Marre, Axel Huebl
License: GPLv3+
"""

import numpy as np
import scipy.constants as const
import scipy.special as scipy

""" @file reference implementation of the rate calculation for bound-bound transitions """


class BoundBoundTransitions:
    @staticmethod
    def _gauntChung(U, cxin1, cxin2, cxin3, cxin4, cxin5):
        """calculate gaunt factor according to,
            H.-K. Chung, R.W. Lee, M.H. Chen,
            A fast Method of to generate collisional exciation cross-sections of highly charged ions a hot dense matter,
            High Energy Density Physics 3 (2007) 342-352

        @param U     float (energy interacting electron)/(delta Energy Transition), unitless
        @param cxin1 float gaunt approximation coefficient 1, unitless
        @param cxin2 float gaunt approximation coefficient 2, unitless
        @param cxin3 float gaunt approximation coefficient 3, unitless
        @param cxin4 float gaunt approximation coefficient 4, unitless
        @param cxin5 float gaunt approximation coefficient 5, unitless

        @return unitless
        """

        return cxin1 * np.log(U) + cxin2 + cxin3 / (U + cxin5) + cxin4 / (U + cxin5) ** 2

    @staticmethod
    def _gauntMewe(U):
        """
        calculate gaunt factor according to,
            R. Mewe,
            Interpolation Formulae for the electron impact exciation of ions in the H-. He-, li- and Ne-Squences,
            Astron. &astrophys. 20, 215-221(1972)

        @param U     float (energy interacting electron)/(delta Energy Transition), unitless

        @return unitless
        """

        return 0.28 * np.log(U) + 0.15

    @staticmethod
    def _multiplicity(levelVector):
        """get degeneracy of atomicState with given levelVector"""
        result = 1.0
        for i, n_i in enumerate(levelVector):
            result *= scipy.comb(2 * (i + 1) ** 2, n_i)
        return result

    @staticmethod
    def collisionalBoundBoundCrossSection(
        energyElectron,
        energyDiffLowerUpper,
        collisionalOscillatorStrength,
        cxin1,
        cxin2,
        cxin3,
        cxin4,
        cxin5,
        lowerStateLevelVector,
        upperStateLevelVector,
        excitation,
    ):
        """cross section for de-/excitation transition due to interaction with free electron

        @param excitation bool true =^= excitation, false =^= deexcitation
        @param energyElectron float energy of interacting free electron, [eV]

        @param energyDiffLowerUpper upperStateEnergy - lowerStateEnergy, [eV]
        @param collisionalOscillatorStrength of the transition, unitless
        @param cxin1 float gaunt approximation coefficient 1, unitless
        @param cxin2 float gaunt approximation coefficient 2, unitless
        @param cxin3 float gaunt approximation coefficient 3, unitless
        @param cxin4 float gaunt approximation coefficient 4, unitless
        @param cxin5 float gaunt approximation coefficient 5, unitless

        @return unit 1e6b, 10^-22 m^2
        """

        U = energyElectron / energyDiffLowerUpper

        # constants
        E_Rydberg = const.physical_constants["Rydberg constant times hc in eV"][0]  # eV
        a0 = const.physical_constants["Bohr radius"][0]  # m
        c0 = 8 * (np.pi * a0) ** 2 / np.sqrt(3.0)  # m^2

        if isinstance(U, np.ndarray) and U.ndim > 0:
            crossSection = np.empty(len(U))

            crossSection[U >= 1.0] = (
                c0
                * (E_Rydberg / energyDiffLowerUpper) ** 2
                * collisionalOscillatorStrength
                * (energyDiffLowerUpper / energyElectron[U >= 1.0])
                / 1e-22
            )  # 1e6b
            crossSection[U < 1.0] = 0.0

        else:
            if U >= 1.0:
                crossSection = (
                    c0
                    * (E_Rydberg / energyDiffLowerUpper) ** 2
                    * collisionalOscillatorStrength
                    * (energyDiffLowerUpper / energyElectron)
                    / 1e-22
                )  # 1e6b
            else:
                crossSection = 0.0

        statisticalRatio = BoundBoundTransitions._multiplicity(
            lowerStateLevelVector
        ) / BoundBoundTransitions._multiplicity(upperStateLevelVector)  # unitless

        if cxin1 == 0 and cxin2 == 0 and cxin3 == 0 and cxin4 == 0 and cxin5 == 0:
            #! @detail Mewe gaunt factor from original FLYCHK publication
            if excitation:
                crossSection *= BoundBoundTransitions._gauntMewe(U)
            else:
                crossSection *= BoundBoundTransitions._gauntMewe(U + 1.0) * statisticalRatio
        else:
            #! @detail chung gaunt factor from 2007 publication if constants are specified
            if excitation:
                crossSection *= BoundBoundTransitions._gauntChung(U, cxin1, cxin2, cxin3, cxin4, cxin5)
            else:
                crossSection *= (
                    BoundBoundTransitions._gauntChung(U, cxin1, cxin2, cxin3, cxin4, cxin5) * statisticalRatio
                )

        return crossSection

    @staticmethod
    def rateCollisionalBoundBoundTransition(
        energyElectron,
        energyElectronBinWidth,
        densityElectrons,
        energyDiffLowerUpper,
        collisionalOscillatorStrength,
        cxin1,
        cxin2,
        cxin3,
        cxin4,
        cxin5,
        lowerStateLevelVector,
        upperStateLevelVector,
        excitation=True,
    ):
        """rate of collisional de-/excitation

        @param excitation bool true =^= excitation, false =^= deexcitation

        @param energyElectron float central energy of electron bin, [eV]
        @param energyElectronBinWidth width of energy bin, [eV]
        @param densityElectrons number density of physical electrons in bin, 1/(m^3 * eV)

        @param energyDiffLowerUpper upperStateEnergy - lowerStateEnergy, [eV]
        @param collisionalOscillatorStrength of the transition, unitless
        @param cxin1 float gaunt approximation coefficient 1
        @param cxin2 float gaunt approximation coefficient 2
        @param cxin3 float gaunt approximation coefficient 3
        @param cxin4 float gaunt approximation coefficient 4
        @param cxin5 float gaunt approximation coefficient 5

        @return unit 1/s
        """

        sigma = BoundBoundTransitions.collisionalBoundBoundCrossSection(
            energyElectron,
            energyDiffLowerUpper,
            collisionalOscillatorStrength,
            cxin1,
            cxin2,
            cxin3,
            cxin4,
            cxin5,
            lowerStateLevelVector,
            upperStateLevelVector,
            excitation,
        )  # 1e6b

        if sigma < 0.0:
            sigma = 0.0

        electronRestMassEnergy = const.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6  # eV

        # derivation for relativist kinetic energy to velocity
        # E = gamma * m*c^2 E_kin + m*c^2 = gamma * m*c^2 => E_kin = (gamma-1) * m*c^2
        # => gamma = E_kin / (m*c^2) + 1
        #  1/sqrt(1-beta^2) = E_kin / (m*c^2) + 1
        #      1/(1-beta^2) = (E_kin/(m*c^2) + 1)^2
        #          1-beta^2 = 1/(E_kin/(m*c^2) + 1)^2
        #            beta^2 = 1 - 1/(E_kin/(m*c^2) + 1)^2
        # => beta = sqrt(1 - 1/(1 + E_kin/(m*c^2))^2)
        # v = c * beta = c * sqrt(1 - 1/(1 + E_kin/(m*c^2))^2)

        # dE * sigma(E) * rho_e * v
        # eV * 1e6b * m^2/(1e6b) * 1/(m^3 * eV) * m/s * unitless
        # = eV/(eV) * m^2 * 1/m^3 * m/s = 1/s
        return (
            energyElectronBinWidth
            * sigma
            * 1e-22
            * densityElectrons
            * const.physical_constants["speed of light in vacuum"][0]
            * np.sqrt(1.0 - 1.0 / (1.0 + energyElectron / electronRestMassEnergy) ** 2)
        )  # 1/s

    @staticmethod
    def rateSpontaneousDeexcitation(
        absorptionOscillatorStrength, frequencyPhoton, lowerStateLevelVector, upperStateLevelVector
    ):
        """rate of spontaneous deexcitation under photon emission

        @param absorptionOscillatorStrength unitless
        @param frequencyPhoton [1/s]
        @param lowerStateLevelVector occupation number level vector for lower state of transition
            i.e. how many electron occupy states in each principal quantum number shell
        @param upperStateLevelVector same as lowerStateLevelVector but upper state of transition

        @return unit 1/s
        """
        # (As)^2 * N/A^2 * / (kg * m/s ) = A^2 * s^2 kg*m/s^2 * 1/A^2 /(kg *m/s )
        # = A^2/A^2 * s^2/s^2 * (kg*m)/(kg*m) * 1/(1/s) = s
        scalingConstant = (
            2
            * np.pi
            * const.physical_constants["elementary charge"][0] ** 2
            * const.physical_constants["vacuum mag. permeability"][0]
            / (const.value("electron mass") * const.physical_constants["speed of light in vacuum"][0])
        )  # s

        ratio = BoundBoundTransitions._multiplicity(lowerStateLevelVector) / BoundBoundTransitions._multiplicity(
            upperStateLevelVector
        )
        # untiless

        # s * 1/s^2 * untiless * unitless = 1/s
        return scalingConstant * frequencyPhoton**2 * ratio * absorptionOscillatorStrength  # 1/s
