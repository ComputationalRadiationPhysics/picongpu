/**
 * Copyright 2013-2016 Rene Widera, Felix Schmitt, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "dataManagement/ISimulationData.hpp"

namespace PMacc
{

    /**
     * Dataset is stored in the DataConnector.
     *
     * It combines simulation data (ISimulationData) with a DatasetStatus which
     * defines if the data should be synchronized.
     *
     */
    class Dataset
    {
    public:

        /**
         * AUTO_OK = data is managed automatically and is already synchronized.\n
         * AUTO_INVALID = data is managed automatically but not up-to-date\n
         */
        enum DatasetStatus
        {
            AUTO_OK, AUTO_INVALID
        };

        /**
         * Returns stored data.
         * Increments the reference counter to this data.
         * This reference has to be released
         * after all read/write operations before the next synchronize()/getData()
         * on this data are done using release().
         *
         * @return reference to internal stored data
         */
        ISimulationData &getData()
        {
            return data;
        }

        /**
         * Constructor
         *
         * @param data reference to simulation data which should be stored
         * @param status initial status for data. AUTO_INVALID by default
         */
        Dataset(ISimulationData &data, DatasetStatus status = AUTO_INVALID) :
        data(data),
        status(status)
        {
        }

        virtual ~Dataset()
        {
        }

        /**
         * Synchronizes stored data if necessary (status == AUTO_INVALID).
         */
        void synchronize()
        {
            if (status == AUTO_INVALID)
            {
                status = AUTO_OK;
                data.synchronize();
            }
        }

        /**
         * Invalidates data synchronization status.
         */
        void invalidate()
        {
            if (this->status == AUTO_OK)
                this->status = AUTO_INVALID;
        }
    private:
        ISimulationData &data;
        DatasetStatus status;
    };
}
