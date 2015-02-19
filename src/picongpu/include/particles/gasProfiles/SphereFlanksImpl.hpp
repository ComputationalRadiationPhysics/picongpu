/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "simulation_defines.hpp"
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct SphereFlanksImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        typedef SphereFlanksImpl<ParamClass> type;
    };

    HINLINE SphereFlanksImpl(uint32_t currentStep)
    {
    }

    /** Calculate the gas density
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        const float_64 unit_length = UNIT_LENGTH;
        const float_X vacuum_y = float_64(ParamClass::vacuum_y_cells) * cellSize.y();
        const floatD_X center = precisionCast<float>(ParamClass::SI().center / unit_length);
        const float_X r = ParamClass::SI::r / unit_length;
        const float_X ri = ParamClass::SI::ri / unit_length;
        const float_X exponent = ParamClass::SI::exponent / unit_length;


        const floatD_X globalCellPos(
                                     precisionCast<float_X>(totalCellOffset) *
                                     cellSize
                                     );

        if (globalCellPos.y() < vacuum_y) return float_X(0.0);

        const float_X radius = math::abs(globalCellPos - center);

        /* "shell": inner radius */
        if (radius < ri)
            return float_X(0.0);
            /* "hard core" */
        else if (radius <= r)
            return float_X(1.0);

            /* "soft exp. flanks"
             *   note: by definition (return, see above) the
             *         argument [ r - radius ] will be element of (-inf, 0) */
        else
            return math::exp((r - radius) * exponent);
    }
};
} //namespace gasProfiles
} //namespace picongpu
