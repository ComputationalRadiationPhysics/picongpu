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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ParticleTags.hpp"

#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics
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
     *          static constexpr char const* ipdIonizationStatesFileName = "";
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
     * - momentum
     * - weighting
     */
    template<typename T_Config>
    struct Ion : public T_Config
    {
        using Tag = Tags::Ion;
    };

    /** indicates species represents an ion in the ionization potential depression calculation (IPD), but does not
     * partake in the atomicPhysics step. (in contrast to Ion)
     * @attention In addition an atomicPhysics ion species also requires the following particle attributes:
     * - momentum
     * - weighting
     */
    struct OnlyIPDIon
    {
        using Tag = Tags::OnlyIPDIon;
    };

    /** indicates species represents electrons and will participate in the atomicPhysics step and the ionization
     * potential depression(IPD) calculation
     *
     * @attention In addition an atomicPhysics ion species also requires the following particle attributes:
     * - momentum
     * - weighting
     */
    struct Electron
    {
        using Tag = Tags::Electron;
    };

    /** indicates species represents electrons in the ionization potential depression calculation(IPD), but does not
     * partake in binning of electrons for the atomicPhysics step. (in contrast to Electron particleType)
     *
     * @attention It is **almost** never a good idea to use this instead of Electron!
     * An electron physically always contributes to both the electron spectrum in the atomicPhyiscs step **and** the
     * IPD calculation. Marking an electron species as onlyIPD unphysically removes part of the electron spectrum for
     * little gain, as the atomicPhysics step scales primarily with the number of ion macro particles.
     *
     * @attention In addition an atomicPhysics ion species also requires the following particle attributes:
     * - momentum
     * - weighting
     */
    struct OnlyIPDElectron
    {
        using Tag = Tags::OnlyIPDElectron;
    };

    namespace traits
    {
        template<typename T_ParticleType, typename T_ParticleTypeTag, typename = void>
        struct IsParticleType : std::false_type
        {
        };

        template<typename T_ParticleType, typename T_ParticleTypeTag>
        struct IsParticleType<
            T_ParticleType,
            T_ParticleTypeTag,
            std::enable_if_t<std::is_same_v<typename T_ParticleType::Tag, T_ParticleTypeTag>>> : std::true_type
        {
        };

        template<typename T_ParticleType, typename T_ParticleTypeTag>
        using IsParticleType_t = typename IsParticleType<T_ParticleType, T_ParticleTypeTag>::type;

        template<typename T_FrameType>
        struct GetParticleType
        {
            using type = typename pmacc::traits::Resolve<
                typename GetFlagType<T_FrameType, atomicPhysicsParticle<>>::type>::type;
        };

        template<typename T_FrameType>
        using GetParticleType_t = typename GetParticleType<T_FrameType>::type;

        template<typename T_Particle, typename T_ParticleTypeTag>
        constexpr bool has([[maybe_unused]] T_ParticleTypeTag const& tag)
        {
            using FrameType = typename std::decay_t<T_Particle>::FrameType;
            constexpr bool hasParticleType = pmacc::traits::HasFlag<FrameType, atomicPhysicsParticle<>>::type::value;

            return hasParticleType && IsParticleType<GetParticleType_t<FrameType>, Tags::Ion>::value;
        }

        template<typename T_MPLSeq, typename T_Tag>
        struct FilterByParticleType
        {
            using SpeciesWithAtomicPhysics =
                typename pmacc::particles::traits::FilterByFlag<T_MPLSeq, atomicPhysicsParticle<>>::type;

            template<typename T_Species>
            using IsTagedType = IsParticleType_t<GetParticleType_t<typename T_Species::FrameType>, T_Tag>;

            using type = pmacc::mp_copy_if<SpeciesWithAtomicPhysics, IsTagedType>;
        };

        template<typename T_MPLSeq, typename T_Tag>
        using FilterByParticleType_t = typename FilterByParticleType<T_MPLSeq, T_Tag>::type;

    } // namespace traits
} // namespace picongpu::particles::atomicPhysics
