/* Copyright 2013-2021 Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
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

#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/Environment.def"
#include "pmacc/communication/ICommunicator.hpp"


namespace pmacc
{
    class EnvironmentController
    {
    public:
        /*! Get communicator
         * @return Communicator for MPI
         */
        ICommunicator& getCommunicator() const
        {
            return *comm;
        }


        /*! Get Mask with all GPU neighbar
         * @return Mask with neighbar
         */
        const Mask& getCommunicationMask() const
        {
            return comm->getCommunicationMask();
        }


        /*! Set MPI communicator
         * @param comm A instance of ICommunicator
         */
        void setCommunicator(ICommunicator& comm)
        {
            this->comm = &comm;
        }

    private:
        friend struct detail::Environment;

        /*! Default constructor.
         */
        EnvironmentController()
        {
        }

        static EnvironmentController& getInstance()
        {
            static EnvironmentController instance;
            return instance;
        }

    private:
        /*! Pointer to MPI communicator.
         */
        ICommunicator* comm;
    };

} // namespace pmacc
