#!/usr/bin/env python
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

""" @file reference implementation of the rate calculation for bound-free collisional transitions """


class BoundFreeCollisionalTransitions:
    @staticmethod
    def _betaFactor(screenedCharge):
        return 0.25 * (np.sqrt((100.0 * screenedCharge + 91.0) / (4.0 * screenedCharge + 3.0)) - 5)  # unitless

    @staticmethod
    def _wFactor(U, screenedCharge):
        return np.power(np.log(U), BoundFreeCollisionalTransitions._betaFactor(screenedCharge) / U)  # unitless

    @staticmethod
    def _multiplicity(lowerStateLevelVector, upperStateLevelVector):
        result = 1.0
        for i in range(len(lowerStateLevelVector)):
            result *= scipy.comb(lowerStateLevelVector[i], lowerStateLevelVector[i] - upperStateLevelVector[i])
        return result

    @staticmethod
    def collisionalIonizationCrossSection(
        energyElectron,
        ionizationEnergy,
        excitationEnergyDifference,
        screenedCharge,
        lowerStateLevelVector,
        upperStateLevelVector,
    ):
        """get cross section for ionization by interaction with a free electron

        @param energyElectron float energy of interacting free electron, [eV]
        @param ionizationEnergy float sum of ionization energies between initial and final charge state, [eV]
        @param excitationEnergyDifference float excitation energy upper state - excitation energy lower state
            all relative to each respective's charge state ground state
        @param screenedCharge screenedCharge of the ionizing shell
        @param lowerStateLevelVector occupation number level vector for lower state of transition
            i.e. how many electron occupy states in each principal quantum number shell
        @param upperStateLevelVector same as lowerStateLevelVector but upper state of transition

        @return unit: 1e6b
        """

        energyDifference = ionizationEnergy + excitationEnergyDifference
        U = energyElectron / energyDifference
        combinatorialFactor = BoundFreeCollisionalTransitions._multiplicity(
            lowerStateLevelVector, upperStateLevelVector
        )

        # m^2 * (eV/(eV))^2 * 1/(eV/eV) * unitless * unitless / (m^2/1e6b) = 1e6b
        if U > 1:
            return (
                np.pi
                * const.value("Bohr radius") ** 2
                * 2.3
                * combinatorialFactor
                * (const.value("Rydberg constant times hc in eV") / energyDifference) ** 2
                * 1.0
                / U
                * np.log(U)
                * BoundFreeCollisionalTransitions._wFactor(U, screenedCharge)
            ) / 1e-22  # 1e6b, 1e-22 m^2
        else:
            return 0.0

    @staticmethod
    def rateCollisionalIonization(
        energyElectron,
        energyElectronBinWidth,
        densityElectrons,
        ionizationEnergy,
        excitationEnergyDifference,
        screenedCharge,
        lowerStateLevelVector,
        upperStateLevelVector,
    ):
        """rate of ionization due to interaction with free electron

        @param energyElectron float central energy of electron bin, [eV]
        @param energyElectronBinWidth float width of energy bin, [eV]
        @param densityElectrons float number density of physical electrons in bin, 1/(m^3 * eV)

        @param ionizationEnergy sum of ionization energies between initial and final charge state, [eV]
        @param excitationEnergyDifference float excitation energy upper state - excitation energy lower state
            all relative to each respective's charge state ground state
        @param screenedCharge screenedCharge of the ionizing shell
        @param lowerStateLevelVector occupation number level vector for lower state of transition
            i.e. how many electron occupy states in each principal quantum number shell
        @param upperStateLevelVector same as lowerStateLevelVector but upper state of transition

        @return unit: 1/s
        """
        sigma = BoundFreeCollisionalTransitions.collisionalIonizationCrossSection(
            energyElectron,
            ionizationEnergy,
            excitationEnergyDifference,
            screenedCharge,
            lowerStateLevelVector,
            upperStateLevelVector,
        )  # 1e6b

        if sigma < 0.0:
            sigma = 0.0

        electronRestMassEnergy = const.value("electron mass energy equivalent in MeV") * 1e6  # eV

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
            * const.value("speed of light in vacuum")
            * np.sqrt(1.0 - 1.0 / (1.0 + energyElectron / electronRestMassEnergy) ** 2)
        )  # 1/s
