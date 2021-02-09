/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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

#include <mpi.h>

#include <vector>
#include <utility>
#include <map>

namespace pmacc
{
    namespace detail
    {
        template<unsigned T_DIM>
        struct LogRankCoords;

        template<>
        struct LogRankCoords<DIM1>
        {
            void operator()(int rank, const int (&coords)[DIM1]) const
            {
                log<ggLog::MPI>("Rank: %1% ; coords %2%") % rank % coords[0];
            }
        };
        template<>
        struct LogRankCoords<DIM2>
        {
            void operator()(int rank, const int (&coords)[DIM2]) const
            {
                log<ggLog::MPI>("Rank: %1% ; coords %2% %3%") % rank % coords[0] % coords[1];
            }
        };
        template<>
        struct LogRankCoords<DIM3>
        {
            void operator()(int rank, const int (&coords)[DIM3]) const
            {
                log<ggLog::MPI>("Rank: %1% ; coords %2% %3% %4%") % rank % coords[0] % coords[1] % coords[2];
            }
        };

    } // namespace detail

    /*! communication via MPI
     */
    template<unsigned DIM>
    class CommunicatorMPI : public ICommunicator
    {
    public:
        /*! ctor
         */
        CommunicatorMPI() : hostRank(0)
        {
            // MPI_Init(nullptr, nullptr);
        }

        /*! dtor
         *
         * calls MPI_Finalize
         */
        virtual ~CommunicatorMPI()
        {
        }

        virtual int getRank()
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

        MPI_Info getMPIInfo() const
        {
            return MPI_INFO_NULL;
        }

        DataSpace<DIM3> getPeriodic() const
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
        void init(DataSpace<DIM3> numberProcesses, DataSpace<DIM3> periodic)
        {
            this->periodic = periodic;

            // check if parameters are correct
            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));

            if(numberProcesses.productOfComponents() != mpiSize)
            {
                throw std::invalid_argument("wrong parameters or wrong mpirun-call!");
            }

            // 1. create Communicator (computing_comm) of computing nodes (ranks 0...n)
            MPI_Comm computing_comm = MPI_COMM_WORLD;

            yoffset = 0;

            // 2. create topology

            dims[0] = numberProcesses.x();
            dims[1] = numberProcesses.y();
            dims[2] = numberProcesses.z();

            topology = MPI_COMM_NULL;

            int periods[] = {periodic.x(), periodic.y(), periodic.z()};

            /*create new communicator based on cartesian coordinates*/
            MPI_CHECK(MPI_Cart_create(computing_comm, DIM, dims, periods, 0, &topology));

            // 3. update Host rank
            updateHostRank();

            // 4. update Coordinates
            updateCoordinates();
        }

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

        virtual const Mask& getCommunicationMask() const
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

        // description in ICommunicator

        MPI_Request* startSend(uint32_t ex, const char* send_data, size_t send_data_count, uint32_t tag)
        {
            MPI_Request* request = new MPI_Request;

            MPI_CHECK(MPI_Isend(
                (void*) send_data,
                static_cast<int>(send_data_count),
                MPI_CHAR,
                ExchangeTypeToRank(ex),
                gridExchangeTag + tag,
                topology,
                request));

            return request;
        }

        // description in ICommunicator

        MPI_Request* startReceive(uint32_t ex, char* recv_data, size_t recv_data_max, uint32_t tag)
        {
            MPI_Request* request = new MPI_Request;

            MPI_CHECK(MPI_Irecv(
                recv_data,
                static_cast<int>(recv_data_max),
                MPI_CHAR,
                ExchangeTypeToRank(ex),
                gridExchangeTag + tag,
                topology,
                request));

            return request;
        }

        // description in ICommunicator

        bool slide()
        {
            // we can only slide in y direction right now
            if(DIM < DIM2)
                return false;

            // MPI_Barrier(topology);
            yoffset--;
            if(yoffset == -dims[1])
                yoffset = 0;

            updateCoordinates();

            return coordinates[1] == dims[1] - 1;
        }

        bool setStateAfterSlides(size_t numSlides)
        {
            // nothing happens
            if(numSlides == 0)
                return false;

            // we can only slide in y direction right now
            if(DIM < DIM2)
                return false;

            bool result = false;

            // only need to apply (numSlides % num-gpus-y) slides
            for(size_t i = 0; i < (numSlides % dims[1]); ++i)
                result = slide();

            return result;
        }


    protected:
        /* Set the first found non charactor or number to 0 (nullptr)
         * name like p1223(Pid=1233) is than p1223
         * in some MPI implementation /mpich) the hostname is unique
         */
        void cleanHostname(char* name)
        {
            for(int i = 0; i < MPI_MAX_PROCESSOR_NAME; ++i)
            {
                if(!(name[i] >= 'A' && name[i] <= 'Z') && !(name[i] >= 'a' && name[i] <= 'z')
                   && !(name[i] >= '0' && name[i] <= '9') && !(name[i] == '_') && !(name[i] == '-'))
                {
                    name[i] = 0;
                    return;
                }
            }
        }

        /*! gets hostRank
         *
         * process with MPI-rank 0 is the master and builds a map with hostname
         * and number of already known processes on this host.
         * Each rank will provide its hostname via MPISend and gets its HostRank
         * from the master.
         *
         */
        void updateHostRank()
        {
            char hostname[MPI_MAX_PROCESSOR_NAME];
            int length;

            MPI_CHECK(MPI_Get_processor_name(hostname, &length));
            cleanHostname(hostname);
            hostname[length++] = '\0';

            MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
            MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));

            if(mpiRank == 0)
            {
                std::map<std::string, int> hosts;
                hosts[hostname] = 0;
                hostRank = 0;
                for(int rank = 1; rank < mpiSize; ++rank)
                {
                    MPI_CHECK(MPI_Recv(
                        hostname,
                        MPI_MAX_PROCESSOR_NAME,
                        MPI_CHAR,
                        rank,
                        gridHostnameTag,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));

                    // printf("Hostname: %s\n", hostname);
                    int hostrank = 0;
                    if(hosts.count(hostname) > 0)
                        hostrank = hosts[hostname] + 1;

                    MPI_CHECK(MPI_Send(&hostrank, 1, MPI_INT, rank, gridHostRankTag, MPI_COMM_WORLD));

                    hosts[hostname] = hostrank;
                }
            }
            else
            {
                MPI_CHECK(MPI_Send(hostname, length, MPI_CHAR, GridManagerRank, gridHostnameTag, MPI_COMM_WORLD));

                MPI_CHECK(MPI_Recv(
                    &hostRank,
                    1,
                    MPI_INT,
                    GridManagerRank,
                    gridHostRankTag,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE));

                // if(hostRank!=0) hostRank--; //!\todo fix mpi hostrank start with 1
            }
        }

        /*! update coordinates \see getCoordinates
         */
        void updateCoordinates()
        {
            // get own coordinates
            int coords[DIM];
            int rank;

            MPI_CHECK(MPI_Comm_rank(topology, &rank));
            MPI_CHECK(MPI_Cart_coords(topology, rank, DIM, coords));

            if(DIM >= DIM2)
            {
                if(dims[1] > 1)
                    coords[1] = (coords[1] + yoffset) % dims[1];

                while(coords[1] < 0)
                    coords[1] += dims[1];
            }

            detail::LogRankCoords<DIM>()(rank, coords);

            for(uint32_t i = 0; i < DIM; ++i)
                this->coordinates[i] = coords[i];

            // init ranks of other hosts
            int mcoords[3];

            communicationMask = Mask();

            for(int i = 1; i < -12 * (int) DIM + 6 * (int) DIM * (int) DIM + 9; i++)
            {
                for(uint32_t j = 0; j < DIM; j++)
                    mcoords[j] = coords[j];

                Mask m(i);
                if(m.containsExchangeType(LEFT))
                    mcoords[0]--;
                if(m.containsExchangeType(RIGHT))
                    mcoords[0]++;

                if(DIM >= DIM2)
                {
                    if(m.containsExchangeType(TOP))
                        mcoords[1]--;
                    if(m.containsExchangeType(BOTTOM))
                        mcoords[1]++;
                }

                if(DIM == DIM3)
                {
                    if(m.containsExchangeType(BACK))
                        mcoords[2]++;
                    if(m.containsExchangeType(FRONT))
                        mcoords[2]--;
                }

                bool ok = true;
                for(uint32_t j = 0; j < DIM; j++)
                    if(periodic[j] == 0
                       && (mcoords[j] < 0
                           || mcoords[j] >= dims[j])) /*only check if no perodic for j dimension is set*/
                        ok = false;

                if(ok)
                {
                    if(dims[1] > 1)
                        mcoords[1] = (mcoords[1] - yoffset) % dims[1];

                    MPI_CHECK(MPI_Cart_rank(topology, mcoords, &ranks[i]));
                    communicationMask = communicationMask + Mask(i);
                }
                else
                {
                    ranks[i] = -1;
                }

                // std::cout << "rank: " << rank << " " << i << " : " << ranks[i] << std::endl;
            }
        }

        /*! converts an exchangeType (e.g. RIGHT) to an MPI-rank
         */
        int ExchangeTypeToRank(uint32_t type)
        {
            return ranks[type];
        }

    private:
        //! coordinates in GPU-Grid [0:cx-1,0:cy-1,0:cz-1]
        DataSpace<DIM> coordinates;

        DataSpace<DIM3> periodic;
        //! MPI communicator (currently MPI_COMM_WORLD)
        MPI_Comm topology;
        //! array for exchangetype-to-rank conversion \see ExchangeTypeToRank
        int ranks[27];
        //! size of pmacc [cx,cy,cz]
        int dims[3];
        //! \see getCommunicationMask
        Mask communicationMask;
        //! rank of this process local to its host (node)
        int hostRank;
        //! offset for sliding window
        int yoffset;

        int mpiRank;
        int mpiSize;
    };

} // namespace pmacc
