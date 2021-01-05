/* Copyright 2016-2021 Heiko Burau
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

#include "pmacc/types.hpp"
#include "pmacc/particles/memory/frames/Frame.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/traits/Resolve.hpp"
#include "pmacc/identifier/alias.hpp"

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
             * typedef bmpl::vector<
             *   particlePusher<UsedParticlePusher>,
             *   shape<UsedParticleShape>,
             *   interpolation<UsedField2Particle>,
             *   current<UsedParticleCurrentSolver>,
             *   massRatio<MassRatioElectrons>,
             *   chargeRatio<ChargeRatioElectrons>,
             *   synchrotronPhotons<PIC_Photons>
             * > ParticleFlagsElectrons;
             *
             * typedef picongpu::Particles<
             *     PMACC_CSTRING( "e" ),
             *     ParticleFlagsElectrons,
             *     DefaultAttributesSeq
             * > PIC_Electrons;
             *
             * typedef typename ResolveAliasFromSpecies<
             *      PIC_Electrons,
             *      synchrotronPhotons<>
             * >::type PhotonSpecies;
             * boost::static_assert(boost::is_same<PhotonsSpecies, PIC_Photons>::value);
             * \endcode
             *
             * \tparam T_SpeciesType particle species
             * \tparam T_Alias alias
             */
            template<typename T_SpeciesType, typename T_Alias>
            struct ResolveAliasFromSpecies;

            template<typename T_SpeciesType, template<typename, typename> class T_Object, typename T_AnyType>
            struct ResolveAliasFromSpecies<T_SpeciesType, T_Object<T_AnyType, pmacc::pmacc_isAlias>>
            {
                typedef T_SpeciesType SpeciesType;
                typedef T_Object<T_AnyType, pmacc::pmacc_isAlias> Alias;
                typedef typename SpeciesType::FrameType FrameType;

                /* The following line only fetches the alias */
                typedef typename pmacc::traits::GetFlagType<FrameType, Alias>::type FoundAlias;

                /* This now resolves the alias into the actual object type */
                typedef typename pmacc::traits::Resolve<FoundAlias>::type type;
            }; // struct ResolveAliasFromSpecies

        } // namespace traits
    } // namespace particles
} // namespace pmacc
