/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/startPosition/OnePositionImpl.def"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.def"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu::particles::startPosition::acc
{
    //! @details shrinks T_ParamClass::inCellOffset to correct length for simDim
    template<typename T_ParamClass>
    struct OnePositionImpl
    {
        /** set in-cell position and weighting
         *
         * @tparam T_Particle pmacc::Particle, particle type
         *
         * @param particle particle to be manipulated
         */
        template<typename T_Particle>
        HDINLINE void operator()(T_Particle& particle)
        {
            constexpr auto initialPosition = T_ParamClass::inCellOffset;
            particle[position_] = initialPosition.template shrink<simDim>();

            // set the weighting attribute if the particle species has it
            constexpr bool hasWeighting
                = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;
            if constexpr(hasWeighting)
                particle[weighting_] = m_weighting;
        }

        template<typename T_Particle>
        HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
        {
            constexpr bool hasWeighting
                = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;

            // note: m_weighting member might stay uninitialized!

            if constexpr(hasWeighting)
            {
                return startPosition::detail::WeightMacroParticles{}(
                    realParticlesPerCell,
                    T_ParamClass::numParticlesPerCell,
                    m_weighting);
            }
            else
            {
                // note: m_weighting member might stay uninitialized!
                return T_ParamClass::numParticlesPerCell;
            }
        }

    private:
        float_X m_weighting;
    };

} // namespace picongpu::particles::startPosition::acc
