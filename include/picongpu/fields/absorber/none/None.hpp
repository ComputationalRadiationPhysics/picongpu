/* Copyright 2021-2023 Sergei Bastrakov
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

#include "picongpu/fields/absorber/Absorber.hpp"

#include <pmacc/Environment.hpp>

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            namespace none
            {
                /** None field absorber implementation
                 *
                 * Does nothing, just checks that all boundaries are periodic.
                 */
                class NoneImpl : public AbsorberImpl
                {
                public:
                    /** Create none absorber implementation instance
                     *
                     * @param cellDescription mapping for kernels
                     */
                    NoneImpl(MappingDesc const cellDescription) : AbsorberImpl(Absorber::Kind::None, cellDescription)
                    {
                        const DataSpace<DIM3> isPeriodicBoundary
                            = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
                        bool areAllBoundariesPeriodic = true;
                        for(uint32_t axis = 0u; axis < simDim; axis++)
                            if(!isPeriodicBoundary[axis])
                                areAllBoundariesPeriodic = false;
                        if(!areAllBoundariesPeriodic)
                            throw std::runtime_error(
                                "None absorber implementation instantiated, but some boundaries are not periodic");
                    }
                };

            } // namespace none
        } // namespace absorber
    } // namespace fields
} // namespace picongpu
