/* Copyright 2013-2019 Rene Widera, Felix Schmitt, Benjamin Worpitz
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

namespace pmacc
{
    /**
     * Interface for a simulation data sorting class.
     *
     * Used for traversing IDs in DataConnector's data container while
     * initialising data (DataConnector.initialise()).
     */
    template<typename ID_TYPE>
    class IDataSorter
    {
    public:
        /**
         * Destructor.
         */
        virtual ~IDataSorter()
        {
        };

        /**
         * Adds an ID to this sorter.
         *
         * @param id data id to add
         */
        virtual void add(ID_TYPE id) = 0;

        /**
         * Returns the first ID for this sorter.
         *
         * @return first id
         */
        virtual ID_TYPE begin() = 0;

        virtual bool isValid() = 0;

        /**
         * Returns if there are more IDs in the sorter.
         *
         * @return whether there are more IDs.
         */
        virtual bool hasNext() = 0;

        /**
         * Returns the next ID in the sorter.
         *
         * Check with hasNext() if there are more IDs.
         *
         * @return next ID
         */
        virtual ID_TYPE getNext() = 0;
    };
}
