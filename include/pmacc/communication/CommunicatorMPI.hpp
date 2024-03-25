/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <utility>

#include <mpi.h>

namespace pmacc
{
    /*! communication via MPI
     */
    template<unsigned DIM>
    class CommunicatorMPI : public ICommunicator
    {
    public:
        CommunicatorMPI() = default;

        virtual ~CommunicatorMPI() = default;

        int getRank() override
        {
            return mpiRank;
        }

        virtual int getSize()
        {
            return mpiSize;
        }

        MPI_Comm getMPIComm() const
        {
            return topology;
        }

        /*! MPI communicator for signal handling
         *
         * @attention Do not use this communicator to transfer simulation data.
         *
         * @return communicator used transfer signal information only
         */
        MPI_Comm getMPISignalComm() const
        {
            return commSignal;
        }

        MPI_Info getMPIInfo() const
        {
            return MPI_INFO_NULL;
        }

        DataSpace<DIM3> getPeriodic() const override
        {
            return this->periodic;
        }

        /*! initializes all processes to build a 3D-grid
         *
         * @param nodes number of GPU nodes in each dimension
         * @param periodic specifying whether the grid is periodic (1) or not (0) in each dimension
         *
         * \warning throws invalid argument if cx*cy*cz != totalnodes
         */
        void init(DataSpace<DIM3> numberProcesses, DataSpace<DIM3> periodic);


        /*! returns a rank number (0-n) for each host
         *
         * E.g. if 8 GPUs are on 2 Hosts (4 GPUs each), the GPUs on each host will get hostrank 0 to 3
         *
         */
        uint32_t getHostRank()
        {
            return hostRank;
        }

        // description in ICommunicator

        const Mask& getCommunicationMask() const override
        {
            return communicationMask;
        }

        /*! returns coordinate of this process in (via init) created grid
         *
         * Coordinates are between [0-cx, 0-cy, 0-cz]
         *
         */
        const DataSpace<DIM> getCoordinates() const
        {
            return this->coordinates;
        }

        //! description in ICommunicator
        MPI_Request* startSend(uint32_t ex, const char* send_data, size_t send_data_count, uint32_t tag) override;


        //! description in ICommunicator
        MPI_Request* startReceive(uint32_t ex, char* recv_data, size_t recv_data_max, uint32_t tag) override;


        //! description in ICommunicator
        bool slide() override;


        bool setStateAfterSlides(size_t numSlides) override;


        /*! converts an exchangeType (e.g. RIGHT) to an MPI-rank
         */
        int ExchangeTypeToRank(uint32_t type)
        {
            return ranks[type];
        }


    protected:
        /* Set the first found non charactor or number to 0 (nullptr)
         * name like p1223(Pid=1233) is than p1223
         * in some MPI implementation /mpich) the hostname is unique
         */
        void cleanHostname(char* name);


        /*! gets hostRank
         *
         * process with MPI-rank 0 is the master and builds a map with hostname
         * and number of already known processes on this host.
         * Each rank will provide its hostname via MPISend and gets its HostRank
         * from the master.
         *
         */
        void updateHostRank();

        /*! update coordinates @see getCoordinates
         */
        void updateCoordinates();

    private:
        //! coordinates in GPU-Grid [0:cx-1,0:cy-1,0:cz-1]
        DataSpace<DIM> coordinates;

        DataSpace<DIM3> periodic;
        //! MPI communicator (currently MPI_COMM_WORLD)
        MPI_Comm topology;
        //! Communicator to handle signals
        MPI_Comm commSignal;
        //! array for exchangetype-to-rank conversion @see ExchangeTypeToRank
        int ranks[27];
        //! size of pmacc [cx,cy,cz]
        int dims[3];
        //! @see getCommunicationMask
        Mask communicationMask;
        //! rank of this process local to its host (node)
        int hostRank{0};
        //! offset for sliding window
        int yoffset;

        int mpiRank;
        int mpiSize;
    };

} // namespace pmacc
