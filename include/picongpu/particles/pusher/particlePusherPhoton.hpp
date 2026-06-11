/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Alexander Grund, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu
{
    namespace particlePusherPhoton
    {
        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                T_FunctorFieldB const functorBField,
                T_FunctorFieldE const functorEField,
                T_Particle& particle,
                T_Pos& pos,
                uint32_t const)
            {
                using MomType = momentum::type;
                MomType const mom = particle[momentum_];

                auto const bField = functorBField(pos);
                auto const eField = functorEField(pos);

                // update probe field if particle contains required attributes
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeB>::type::value)
                    particle[probeB_] = bField;
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeE>::type::value)
                    particle[probeE_] = eField;

                float_X const normMom = pmacc::math::l2norm(mom);
                MomType const vel = mom * (sim.pic.getSpeedOfLight() / normMom);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    pos[d] += (vel[d] * sim.pic.getDt()) / sim.pic.getCellSize()[d];
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "other");
                propList["param"] = "free streaming photon pusher";
                return propList;
            }
        };
    } // namespace particlePusherPhoton
} // namespace picongpu
