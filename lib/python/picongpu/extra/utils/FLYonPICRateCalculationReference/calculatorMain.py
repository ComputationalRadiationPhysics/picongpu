#!/usr/bin/env python
"""
atomicPhysics(FLYonPIC) reference rate calculation
This file is part of the PIConGPU.
Copyright 2023-2024 PIConGPU contributors
Authors: Brian Marre
License: GPLv3+
"""

import BoundBoundTransitions as boundbound
import BoundFreeCollisionalTransitions as boundfreecollisional
import BoundFreeFieldTransitions as boundfreefield

import scipy.constants as const
import numpy as np
import mpmath as mp

""" @file call to get print out of reference rates/crossections """

if __name__ == "__main__":
    # electron histogram info
    # eV
    energyElectron = 1000.0
    # eV
    energyElectronBinWidth = 10.0
    # 1/(eV * m^3)
    densityElectrons = 1e28

    # bound-free collisional transition data
    # eV
    ionizationEnergy = 100.0
    # eV
    excitationEnergyDifference = 5.0
    # e
    screenedCharge = 5.0
    lowerStateLevelVectorBoundFree = (1, 1, 0, 0, 0, 0, 1, 0, 1, 0)
    upperStateLevelVectorBoundFree = (1, 1, 0, 0, 0, 0, 1, 0, 0, 0)

    # bound-free field transition data
    #   screened Charge lower state - 1
    screenedCharge = 4
    # in eV
    Hartree = 27.211386245981
    # in Hartree
    ionizationEnergy = 5.0 / Hartree
    # in atomic_unit["electric field"] ~ 5.1422e11 V/m
    fieldStrength = 0.0126

    # bound-bound transition data
    # eV
    energyDiffLowerUpper = 5.0
    cxin1 = 1.0
    cxin2 = 2.0
    cxin3 = 3.0
    cxin4 = 4.0
    cxin5 = 5.0
    collisionalOscillatorStrength = 1.0
    absorptionOscillatorStrength = 1.0e-1
    lowerStateLevelVectorBoundBound = (1, 0, 2, 0, 0, 0, 1, 0, 0, 0)
    upperStateLevelVectorBoundBound = (1, 0, 1, 0, 0, 0, 1, 0, 1, 0)
    # 1/s
    frequencyPhoton = energyDiffLowerUpper / const.physical_constants["Planck constant in eV/Hz"][0]

    print("cross sections:")
    print("- bound-free")
    print(
        "\t collisional ionization cross section: \t  {0:.12e} 1e6*barn".format(
            boundfreecollisional.BoundFreeCollisionalTransitions.collisionalIonizationCrossSection(
                energyElectron,
                ionizationEnergy,
                excitationEnergyDifference,
                screenedCharge,
                lowerStateLevelVectorBoundFree,
                upperStateLevelVectorBoundFree,
            )
        )
    )

    print("- bound-bound")
    print(
        "\t collisional excitation cross section: \t  {0:.12e} 1e6*barn".format(
            boundbound.BoundBoundTransitions.collisionalBoundBoundCrossSection(
                energyElectron,
                energyDiffLowerUpper,
                collisionalOscillatorStrength,
                cxin1,
                cxin2,
                cxin3,
                cxin4,
                cxin5,
                lowerStateLevelVectorBoundBound,
                upperStateLevelVectorBoundBound,
                excitation=True,
            )
        )
    )
    print(
        "\t collisional deexcitation cross section:  {0:.12e} 1e6*barn".format(
            boundbound.BoundBoundTransitions.collisionalBoundBoundCrossSection(
                energyElectron,
                energyDiffLowerUpper,
                collisionalOscillatorStrength,
                cxin1,
                cxin2,
                cxin3,
                cxin4,
                cxin5,
                lowerStateLevelVectorBoundBound,
                upperStateLevelVectorBoundBound,
                excitation=False,
            )
        )
    )

    print("rates:")
    print("- bound-free collsional")
    print(
        "\t collisional ionization rate:  \t\t  {0:.12e} 1/s".format(
            boundfreecollisional.BoundFreeCollisionalTransitions.rateCollisionalIonization(
                energyElectron,
                energyElectronBinWidth,
                densityElectrons,
                ionizationEnergy,
                excitationEnergyDifference,
                screenedCharge,
                lowerStateLevelVectorBoundFree,
                upperStateLevelVectorBoundFree,
            )
        )
    )

    print("- bound-free field")
    # in unit electric field atomic units
    nEff = boundfreefield.BoundFreeFieldTransitions.n_eff_numpy(screenedCharge, ionizationEnergy)
    fieldStrengthMaxADKRate = 4.0 * screenedCharge**3 / (3.0 * nEff**3 * (4.0 * nEff - 3.0))
    print(
        "\t ADK fieldStrength: {0:.4e}, F_maxADK: {1:.4e}, F_crit_BSI: {2:.4e}".format(
            fieldStrength,
            fieldStrengthMaxADKRate,
            boundfreefield.BoundFreeFieldTransitions.F_crit_BSI(screenedCharge, ionizationEnergy),
        )
    )

    print(
        "\t ADK rate(numpy) : {0:.9e} * 1/(3.3e-17s)".format(
            np.float32(
                boundfreefield.BoundFreeFieldTransitions.ADKRate_numpy(
                    np.float32(screenedCharge), np.float32(ionizationEnergy), np.float32(fieldStrength)
                )
                * 3.3e-17
                / boundfreefield.atomic_unit["time"]
            )
        )
    )
    print(
        "\t ADK rate(mpmath): "
        + mp.nstr(
            boundfreefield.BoundFreeFieldTransitions.ADKRate_mpmath(
                mp.mpf(screenedCharge), mp.mpf(ionizationEnergy), mp.mpf(fieldStrength)
            )
            * mp.mpf(3.3e-17)
            / mp.mpf(boundfreefield.atomic_unit["time"]),
            10,
            max_fixed=1,
        )
        + " * 1/(3.3e-17s)"
    )

    print("- bound-bound")
    print(
        "\t collisional excitation rate:  \t\t  {0:.12e} 1/s".format(
            boundbound.BoundBoundTransitions.rateCollisionalBoundBoundTransition(
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
                lowerStateLevelVectorBoundBound,
                upperStateLevelVectorBoundBound,
                excitation=True,
            )
        )
    )
    print(
        "\t collisional deexcitation rate:  \t  {0:.12e} 1/s".format(
            boundbound.BoundBoundTransitions.rateCollisionalBoundBoundTransition(
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
                lowerStateLevelVectorBoundBound,
                upperStateLevelVectorBoundBound,
                excitation=False,
            )
        )
    )
    print(
        "\t spontaneous radiative deexcitation rate: {0:.12e} 1/s".format(
            boundbound.BoundBoundTransitions.rateSpontaneousDeexcitation(
                absorptionOscillatorStrength,
                frequencyPhoton,
                lowerStateLevelVectorBoundBound,
                upperStateLevelVectorBoundBound,
            )
        )
    )
