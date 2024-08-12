/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/identifier/alias.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::particleType
{
    /** indicates species represents an ion and will participate in atomicPhysics step and
     * ionization potential depression calculation
     *
     * @tparam T_Config configuration required by ion species partaking in the atomicPhysics step
     *
     * @example T_Config for Argon in tp be defined for every atomic physics ion species in `speciesDefinition.param`
     *   @code{.cpp}
     *      struct ArgonAtomicPhysicsCfg
     *      {
     *          //! paths to atomic input data files, not provided with PIConGPU, see atomicPhysics documentation
     *          static constexpr char const* chargeStatesFileName = "./ChargeStates_Ar.txt";
     *          static constexpr char const* atomicStatesFileName = "./AtomicStates_Ar.txt";
     *
     *          static constexpr char const* boundBoundTransitionsFileName = "./BoundBoundTransitions_Ar.txt";
     *          static constexpr char const* boundFreeTransitionsFileName = "./BoundFreeTransitions_Ar.txt";
     *          static constexpr char const* autonomousTransitionsFileName = "./AutonomousTransitions_Ar.txt";
     *
     *          static constexpr char const* pressureIonizationStatesFileName = "";
     *
     *          //! configuration of atomic data storage and active processes in atomicPhysics
     *          using AtomicDataType = atomicPhysics::AtomicData_Ar;
     *
     *          //! number of atomic states in input file
     *          static constexpr uint16_t numberAtomicStates = 470u;
     *
     *          //! species of ionization electrons
     *          using IonizationElectronSpecies = BulkElectrons;
     *      };
     *
     *      Ion<ArgonAtomicPhysicsCfg>;
     *   @endcode
     *
     * @attention In addition an atomicPhysics ion species also requires the following particle attributes:
     * - atomicStateCollectionIndex
     * - processClass
     * - transitionIndex
     * - binIndex
     * - accepted
     */
    alias(Ion);

    /** indicates species represents an ion in the ionization potential depression calculation (IPD), but does not
     * partake in the atomicPhysics step. (in contrast to Ion)
     */
    struct OnlyIPDIon
    {
    };

    /** indicates species represents electrons and will participate in the atomicPhysics step and the ionization
     * potential depression(IPD) calculation
     */
    struct Electron
    {
    };

    /** indicates species represents electrons in the ionization potential depression calculation(IPD), but does not
     * partake in binning of electrons for the atomicPhysics step. (in contrast to Electron particleType)
     *
     * @attention It is **almost** never a good idea to use this instead of Electron!
     * An electron physically always contributes to both the electron spectrum in the atomicPhyiscs step **and** the
     * IPD calculation. Marking an electron species as onlyIPD unphysically removes part of the electron spectrum for
     * little gain, as the atomicPhysics step scales primarily with the number of ion macro particles.
     */
    struct OnlyIPDElectron
    {
    };
} // namespace picongpu::particles::atomicPhysics::particleType
