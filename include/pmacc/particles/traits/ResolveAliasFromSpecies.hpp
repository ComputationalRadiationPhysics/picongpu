/* Copyright 2016-2023 Heiko Burau
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
