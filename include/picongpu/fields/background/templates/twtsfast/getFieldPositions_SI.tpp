/* Copyright 2014-2021 Alexander Debus
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

#include <pmacc/types.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include "picongpu/fields/background/templates/twtsfast/numComponents.hpp"

namespace picongpu
{
    namespace templates
    {
        namespace twtsfast
        {
            /** Auxiliary functions for calculating the TWTS field */
            namespace detail
            {
                /** Calculate the SI position vectors that later enter the Ex(r, t), By(r, t)
                 *  and Bz(r ,t) calculations as r.
                 *  @param cellIdx The total cell id counted from the start at timestep 0. */
                HDINLINE pmacc::math::Vector<floatD_64, numComponents> getFieldPositions_SI(
                    DataSpace<simDim> const& cellIdx,
                    DataSpace<simDim> const& halfSimSize,
                    pmacc::math::Vector<floatD_X, numComponents> const& fieldOnGridPositions,
                    float_64 const unit_length,
                    float_64 const focus_y_SI,
                    float_X const phi)
                {
                    /* Note: Neither direct precisionCast on picongpu::cellSize
                       or casting on floatD_ does work. */
                    floatD_64 const cellDim(picongpu::cellSize.shrink<simDim>());
                    floatD_64 const cellDimensions = cellDim * unit_length;

                    /* TWTS laser coordinate origin is centered transversally and defined longitudinally by
                       the laser center in y (usually maximum of intensity). */
                    floatD_X laserOrigin = precisionCast<float_X>(halfSimSize);
                    laserOrigin.y() = float_X(focus_y_SI / cellDimensions.y());

                    /* For staggered fields (e.g. Yee-grid), obtain the fractional cell index components and add
                     * that to the total cell indices. The physical field coordinate origin is transversally
                     * centered with respect to the global simulation volume.
                     * pmacc::math::Vector<floatD_X, numComponents> fieldPositions =
                     *                traits::FieldPosition<fields::CellType, FieldE>( ); */
                    pmacc::math::Vector<floatD_X, numComponents> fieldPositions = fieldOnGridPositions;

                    pmacc::math::Vector<floatD_64, numComponents> fieldPositions_SI;

                    for(uint32_t i = 0; i < numComponents; ++i) /* cellIdx Ex, Ey and Ez */
                    {
                        fieldPositions[i] += (precisionCast<float_X>(cellIdx) - laserOrigin);
                        fieldPositions_SI[i] = precisionCast<float_64>(fieldPositions[i]) * cellDimensions;

                        fieldPositions_SI[i] = rotateField(fieldPositions_SI[i], phi);
                    }

                    return fieldPositions_SI;
                }

            } /* namespace detail */
        } /* namespace twtsfast */
    } /* namespace templates */
} /* namespace picongpu */
