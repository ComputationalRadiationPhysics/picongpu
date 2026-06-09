/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct CylindricalImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = CylindricalImpl<ParamClass>;
            };

            HINLINE CylindricalImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                floatD_X const globalCellPosSimDim(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());


                auto const globalCellPos = [&]
                {
                    if constexpr(simDim == 2)
                    {
                        return float3_X{globalCellPosSimDim[0], globalCellPosSimDim[1], 0.0_X};
                    }
                    else
                    {
                        return globalCellPosSimDim;
                    }
                }();
                constexpr float3_X centerPosition
                    = precisionCast<float_X>(ParamClass::centerPosition_SI / sim.unit.length());
                float3_X const cellPositionRel = globalCellPos - centerPosition;
                constexpr float3_X jetAxis = ParamClass::jetAxis;
                float_X const r = math::l2norm(math::cross(cellPositionRel, jetAxis)) / math::l2norm(jetAxis);


                constexpr float_X radius = static_cast<float_X>(ParamClass::radius_SI / sim.unit.length());

                float_X density = 0.0;
                // prePlasma ramp
                if constexpr(ParamClass::prePlasmaLength_SI == 0.0 || ParamClass::prePlasmaCutoff_SI == 0.0)
                {
                    density = (r < radius) ? 1.0 : 0.0;
                }
                else
                {
                    constexpr float_X prePlasmaLength
                        = precisionCast<float_X>(ParamClass::prePlasmaLength_SI / sim.unit.length());
                    constexpr float_X prePlasmaCutoff
                        = precisionCast<float_X>(ParamClass::prePlasmaCutoff_SI / sim.unit.length());
                    float_X const reduced_radius
                        = math::sqrt(radius * radius - prePlasmaLength * prePlasmaLength) - prePlasmaLength;

                    if(r < reduced_radius)
                    {
                        density = 1.0;
                    }
                    else if(r < (reduced_radius + prePlasmaCutoff))
                    {
                        float_X const distance = r - reduced_radius;
                        density = math::exp(-distance / prePlasmaLength);
                    }
                }
                density *= ParamClass::densityFactor;
                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
