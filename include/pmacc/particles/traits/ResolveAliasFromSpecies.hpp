/*
 * SPDX-FileCopyrightText: Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/identifier/alias.hpp"
#include "pmacc/particles/memory/frames/Frame.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/traits/Resolve.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace traits
        {
            /** Resolves a custom alias in the flag list of a particle species.
             *
             * Example:
             *
             * \code{.cpp}
             * typedef mp_list<
             *   particlePusher<UsedParticlePusher>,
             *   shape<UsedParticleShape>,
             *   interpolation<UsedField2Particle>,
             *   current<UsedParticleCurrentSolver>,
             *   massRatio<MassRatioElectrons>,
             *   chargeRatio<ChargeRatioElectrons>,
             * > ParticleFlagsElectrons;
             *
             * typedef picongpu::Particles<
             *     PMACC_CSTRING( "e" ),
             *     ParticleFlagsElectrons,
             *     DefaultAttributesSeq
             * > PIC_Electrons;
             *
             * using InterpolationMethod = typename ResolveAliasFromSpecies<
             *      PIC_Electrons,
             *      interpolation<>
             * >::type;
             * boost::static_assert(boost::is_same<InterpolationMethod, UsedField2Particle>::value);
             * \endcode
             *
             * @tparam T_SpeciesType particle species
             * @tparam T_Alias alias
             */
            template<typename T_SpeciesType, typename T_Alias>
            struct ResolveAliasFromSpecies;

            template<typename T_SpeciesType, template<typename, typename> class T_Object, typename T_AnyType>
            struct ResolveAliasFromSpecies<T_SpeciesType, T_Object<T_AnyType, pmacc::pmacc_isAlias>>
            {
                using SpeciesType = T_SpeciesType;
                using Alias = T_Object<T_AnyType, pmacc::pmacc_isAlias>;
                using FrameType = typename SpeciesType::FrameType;

                /* The following line only fetches the alias */
                using FoundAlias = typename pmacc::traits::GetFlagType<FrameType, Alias>::type;

                /* This now resolves the alias into the actual object type */
                using type = typename pmacc::traits::Resolve<FoundAlias>::type;
            }; // struct ResolveAliasFromSpecies

        } // namespace traits
    } // namespace particles
} // namespace pmacc
