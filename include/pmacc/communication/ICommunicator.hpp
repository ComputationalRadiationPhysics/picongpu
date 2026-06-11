/*
 * SPDX-FileCopyrightText: Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <mpi.h>

namespace pmacc
{
    /*! Interface for communication
     */
    class ICommunicator
    {
    public:
        /*! returns available communication partners
         *
         * returns a mask with neighbors, e.g. if there is a right neighbor result.isSet(RIGHT) returns true
         */
        virtual Mask const& getCommunicationMask() const = 0;

        /*! moves all GPUs from top to bottom (y-coordinate)
         *
         * @return true if the position of gpu is switched to the end, else false
         */
        virtual bool slide() = 0;

        /*! slides multiple times
         *
         * @param[in] numSlides number of slides
         * @return true if the position of gpu is switched to the end, else false
         */
        virtual bool setStateAfterSlides(size_t numSlides) = 0;

        //!\todo Interface should not depend on MPI!

        /*! starts sending via MPI (non-blocking)
         *
         * @param[in] ex                direction to send (enum ExchangeType)
         * @param[in] send_data         pointer to data; should have at least send_data_count bytes
         * @param[in] send_data_count   message size in bytes to sent
         * @param[in] tag               user-defined tag; only message with the same tag can be exchanged (i.e.
         * startSend and startReceive must use the same tag) @returns an request for testing if this operation has
         * already finished
         */
        virtual MPI_Request* startSend(uint32_t ex, char const* send_data, size_t send_data_count, uint32_t tag) = 0;

        /*! starts receiving via MPI (non-blocking)
         *
         * If recv_data_max is less then send_data_count (on other host) multiple startReceive are needed!
         *
         * @param[in] ex                direction to send (enum ExchangeType)
         * @param[in] recv_data         pointer to data; should have at least recv_data_max bytes
         * @param[in] recv_data_max     maximum message size in bytes to receive
         * @param[in] tag               user-defined tag; only message with the same tag can be exchanged (i.e.
         * startSend and startReceive must use the same tag) @returns an request for testing if this operation has
         * already finished
         */
        virtual MPI_Request* startReceive(uint32_t ex, char* recv_data, size_t recv_data_max, uint32_t tag) = 0;

        virtual int getRank() = 0;

        /*! Return which of the three directions are periodic
         *
         * @return for each direction a false (0) or true(1) value
         */
        virtual DataSpace<DIM3> getPeriodic() const = 0;
    };

} // namespace pmacc
