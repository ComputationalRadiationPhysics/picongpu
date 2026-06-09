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
#include "picongpu/simulation/control/MovingWindow.hpp"

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct SphereFlanksImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = SphereFlanksImpl<ParamClass>;
            };

            HINLINE SphereFlanksImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                float_64 const unit_length = sim.unit.length();
                float_X const vacuum_y = float_X(ParamClass::vacuumCellsY) * sim.pic.getCellSize().y();
                floatD_X const center = precisionCast<float_32>(ParamClass::center_SI / unit_length);
                float_X const r = ParamClass::r_SI / unit_length;
                float_X const ri = ParamClass::ri_SI / unit_length;
                float_X const exponent = ParamClass::exponent_SI * unit_length;


                floatD_X const globalCellPos(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());

                if(globalCellPos.y() < vacuum_y)
                    return float_X(0.0);

                float_X const distance = pmacc::math::l2norm(globalCellPos - center);

                /* "shell": inner radius */
                if(distance < ri)
                    return float_X(0.0);
                /* "hard core" */
                else if(distance <= r)
                    return float_X(1.0);

                /* "soft exp. flanks"
                 *   note: by definition (return, see above) the
                 *         argument [ r - distance ] will be element of (-inf, 0) */
                else
                    return math::exp((r - distance) * exponent);
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
