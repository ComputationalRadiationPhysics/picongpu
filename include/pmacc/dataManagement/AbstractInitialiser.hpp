/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/dataManagement/ISimulationData.hpp"


namespace pmacc
{
    /**
     * Abstract base class for initialising simulation data (ISimulationData).
     */
    class AbstractInitialiser
    {
    public:
        /**
         * Setup this initialiser.
         * Called before any init.
         *
         * @return the next timestep
         */
        virtual uint32_t setup()
        {
            return 0;
        };

        /**
         * Tears down this initialiser.
         * Called after any init.
         */
        virtual void teardown(){};

        /**
         * Initialises simulation data (concrete type of data is described by id).
         *
         * @param data reference to actual simulation data
         * @param currentStep current simulation iteration
         */
        virtual void init(ISimulationData& data, uint32_t currentStep) = 0;
    };

} // namespace pmacc
