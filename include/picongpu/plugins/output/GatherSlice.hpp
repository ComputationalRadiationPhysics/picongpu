/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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
#include "picongpu/plugins/output/header/MessageHeader.hpp"

#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/types.hpp>

#include <mpi.h>

#include <vector>
#include <sys/stat.h>


namespace picongpu
{
    using namespace pmacc;

    struct GatherSlice
    {
        GatherSlice()
            : mpiRank(-1)
            , numRanks(0)
            , filteredData(nullptr)
            , comm(MPI_COMM_NULL)
            , fullData(nullptr)
            , masterRank(0)
            , isMPICommInitialized(false)
        {
        }

        ~GatherSlice()
        {
            reset();
        }

        /*
         * @return true if object has reduced data after reduce call else false
         */
        bool init(bool isActive)
        {
            static int masterRankOffset = 0;

            /* free old communicator if `init()` is called again */
            if(isMPICommInitialized)
            {
                reset();
            }

            int countRanks = Environment<simDim>::get().GridController().getGpuNodes().productOfComponents();
            std::vector<int> gatherRanks(countRanks);
            std::vector<int> groupRanks(countRanks);
            mpiRank = Environment<simDim>::get().GridController().getGlobalRank();
            if(!isActive)
                mpiRank = -1;

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Allgather(&mpiRank, 1, MPI_INT, &gatherRanks[0], 1, MPI_INT, MPI_COMM_WORLD));

            for(int i = 0; i < countRanks; ++i)
            {
                if(gatherRanks[i] != -1)
                {
                    groupRanks[numRanks] = gatherRanks[i];
                    numRanks++;
                }
            }

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_Group group = MPI_GROUP_NULL;
            MPI_Group newgroup = MPI_GROUP_NULL;
            MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
            MPI_CHECK(MPI_Group_incl(group, numRanks, &groupRanks[0], &newgroup));

            MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &comm));

            if(mpiRank != -1)
            {
                MPI_Comm_rank(comm, &mpiRank);
                isMPICommInitialized = true;
            }
            MPI_CHECK(MPI_Group_free(&group));
            MPI_CHECK(MPI_Group_free(&newgroup));

            masterRankOffset++;
            /* avoid that only rank zero is the master
             * this reduces the load of rank zero
             */
            masterRank = (masterRankOffset % numRanks);

            return mpiRank == masterRank;
        }

        template<class Box>
        Box operator()(Box& data, const MessageHeader& header)
        {
            using ValueType = typename Box::ValueType;

            Box dstBox = Box(PitchedBox<ValueType, DIM2>(
                (ValueType*) filteredData,
                DataSpace<DIM2>(),
                header.sim.size,
                header.sim.size.x() * sizeof(ValueType)));

            MessageHeader* fakeHeader = MessageHeader::create();
            *fakeHeader = header;

            char* recvHeader = new char[MessageHeader::bytes * numRanks];

            if(fullData == nullptr && mpiRank == masterRank)
                fullData = (char*) new ValueType[header.sim.size.productOfComponents()];


            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Gather(
                fakeHeader,
                MessageHeader::bytes,
                MPI_CHAR,
                recvHeader,
                MessageHeader::bytes,
                MPI_CHAR,
                masterRank,
                comm));

            std::vector<int> counts(numRanks);
            std::vector<int> displs(numRanks);
            int offset = 0;
            for(int i = 0; i < numRanks; ++i)
            {
                MessageHeader* head = (MessageHeader*) (recvHeader + MessageHeader::bytes * i);
                counts[i] = head->node.maxSize.productOfComponents() * sizeof(ValueType);
                displs[i] = offset;
                offset += counts[i];
            }

            const size_t elementsCount = header.node.maxSize.productOfComponents() * sizeof(ValueType);

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Gatherv(
                (char*) (data.getPointer()),
                elementsCount,
                MPI_CHAR,
                fullData,
                &counts[0],
                &displs[0],
                MPI_CHAR,
                masterRank,
                comm));


            if(mpiRank == masterRank)
            {
                log<picLog::DOMAINS>("Master create image");
                if(filteredData == nullptr)
                    filteredData = (char*) new ValueType[header.sim.size.productOfComponents()];

                /*create box with valid memory*/
                dstBox = Box(PitchedBox<ValueType, DIM2>(
                    (ValueType*) filteredData,
                    DataSpace<DIM2>(),
                    header.sim.size,
                    header.sim.size.x() * sizeof(ValueType)));

                for(int i = 0; i < numRanks; ++i)
                {
                    MessageHeader* head = (MessageHeader*) (recvHeader + MessageHeader::bytes * i);

                    log<picLog::DOMAINS>("part image with offset %1%byte=%2%elements | size %3%  | offset %4%")
                        % displs[i] % (displs[i] / sizeof(ValueType)) % head->node.maxSize.toString()
                        % head->node.offset.toString();
                    Box srcBox = Box(PitchedBox<ValueType, DIM2>(
                        (ValueType*) (fullData + displs[i]),
                        DataSpace<DIM2>(),
                        head->node.maxSize,
                        head->node.maxSize.x() * sizeof(ValueType)));

                    insertData(dstBox, srcBox, head->node.offset, head->node.maxSize);
                }

                __deleteArray(fullData);
            }

            delete[] recvHeader;
            MessageHeader::destroy(fakeHeader);

            return dstBox;
        }

        template<class DstBox, class SrcBox>
        void insertData(
            DstBox& dst,
            const SrcBox& src,
            MessageHeader::Size2D offsetToSimNull,
            MessageHeader::Size2D srcSize)
        {
            for(int y = 0; y < srcSize.y(); ++y)
            {
                for(int x = 0; x < srcSize.x(); ++x)
                {
                    dst[y + offsetToSimNull.y()][x + offsetToSimNull.x()] = src[y][x];
                }
            }
        }

    private:
        /*reset this object und set all values to initial state*/
        void reset()
        {
            mpiRank = -1;
            numRanks = 0;
            if(filteredData != nullptr)
                delete[] filteredData;
            filteredData = nullptr;
            if(fullData != nullptr)
                delete[] fullData;
            fullData = nullptr;
            if(isMPICommInitialized)
            {
                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Comm_free(&comm));
            }
            isMPICommInitialized = false;
        }

        char* filteredData;
        char* fullData;
        MPI_Comm comm;
        int mpiRank;
        int numRanks;
        int masterRank;
        bool isMPICommInitialized;
    };

} // namespace picongpu
