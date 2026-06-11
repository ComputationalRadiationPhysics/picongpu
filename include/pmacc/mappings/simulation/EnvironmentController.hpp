/*
 * SPDX-FileCopyrightText: Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.def"
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"

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
        Mask const& getCommunicationMask() const
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
        EnvironmentController() = default;

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
