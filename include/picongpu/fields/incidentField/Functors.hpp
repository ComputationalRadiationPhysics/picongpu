/* Copyright 2020-2022 Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            //! Concept defining interface of incident field functors for E and B
            struct FunctorIncidentFieldConcept
            {
                /** Create a functor
                 *
                 * @param unitField conversion factor from SI to internal units,
                 *                  field_internal = field_SI / unitField
                 */
                HDINLINE FunctorIncidentFieldConcept(float3_64 unitField);

                /** Return incident field for the given position and time.
                 *
                 * Note that component matching the source boundary (e.g. the y component for YMin and YMax boundaries)
                 * will not be used, and so should be zero.
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 * @param currentStep current time step index, note that it is fractional
                 * @return incident field value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx, float_X currentStep) const;
            };

            //! Helper incident field functor always returning 0
            struct ZeroFunctor
            {
                /** Create a functor
                 *
                 * @param unitField conversion factor from SI to internal units,
                 *                  field_internal = field_SI / unitField
                 */
                HDINLINE ZeroFunctor(float3_64 unitField)
                {
                }

                /** Return zero incident field for any given position and time.
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 * @param currentStep current time step index, note that it is fractional
                 * @return incident field value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx, float_X currentStep) const
                {
                    return float3_X::create(0.0_X);
                }
            };


        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
