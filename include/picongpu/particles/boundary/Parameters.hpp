/* Copyright 2021 Lennert Sprenger
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
    namespace particles
    {
        namespace boundary
        {
            //! Parameters to be passed to particle boundary conditions
            struct Parameters
            {
                /** Begin of the internal (relative to boundary) cells in total coordinates
                 *
                 * Particles to the left side are outside
                 */
                pmacc::DataSpace<simDim> beginInternalCellsTotal;

                /** End of the internal (relative to boundary) cells in total coordinates
                 *
                 * Particles equal or to the right side are outside
                 */
                pmacc::DataSpace<simDim> endInternalCellsTotal;
            };
        } // namespace boundary
    } // namespace particles
} // namespace picongpu
