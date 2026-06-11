/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/absorber/AbsorberImpl.hpp"

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
                        DataSpace<DIM3> const isPeriodicBoundary
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
