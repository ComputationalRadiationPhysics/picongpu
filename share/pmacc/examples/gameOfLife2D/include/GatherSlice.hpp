/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Maximilian Knespel, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/types.hpp> // DIM*

#include <mpi.h>

namespace gol
{
    using namespace pmacc;

    struct MessageHeader
    {
        MessageHeader()
        {
        }

        MessageHeader(Space simSize, GridLayout<DIM2> layout, Space nodeOffset)
            : simSize(simSize)
            , nodeOffset(nodeOffset)
        {
            nodeSize = layout.getDataSpace();
            nodePictureSize = layout.getDataSpaceWithoutGuarding();
            nodeGuardCells = layout.getGuard();
        }

        Space simSize;
        Space nodeSize;
        Space nodePictureSize;
        Space nodeGuardCells;
        Space nodeOffset;
    };

    struct GatherSlice
    {
        GatherSlice() : mpiRank(-1), numRanks(0), filteredData(nullptr), fullData(nullptr), isMPICommInitialized(false)
        {
        }

        ~GatherSlice()
        {
        }

        void finalize()
        {
            if(filteredData != nullptr)
            {
                delete[] filteredData;
                filteredData = nullptr;
            }
            if(fullData != nullptr)
            {
                delete[] fullData;
                fullData = nullptr;
            }
            if(isMPICommInitialized)
            {
                MPI_Comm_free(&comm);
                isMPICommInitialized = false;
            }
            mpiRank = -1;
        }

        /*
         * Saves the message header and creates a new MPI group with all ranks
         * that called this with isActive = true
         * @return true if the current rank is the master of the new MPI group
         */
        bool init(const MessageHeader mHeader, bool isActive)
        {
            header = mHeader;

            int countRanks = Environment<DIM2>::get().GridController().getGpuNodes().productOfComponents();
            std::vector<int> gatherRanks(countRanks);
            std::vector<int> groupRanks(countRanks);
            mpiRank = Environment<DIM2>::get().GridController().getGlobalRank();
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
            MPI_Group group;
            MPI_Group newgroup;
            MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
            MPI_CHECK(MPI_Group_incl(group, numRanks, &groupRanks[0], &newgroup));

            MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &comm));

            if(mpiRank != -1)
            {
                MPI_Comm_rank(comm, &mpiRank);
                isMPICommInitialized = true;
            }

            return mpiRank == 0;
        }

        template<class Box>
        Box operator()(Box data)
        {
            typedef typename Box::ValueType ValueType;

            Box dstBox = Box(PitchedBox<ValueType, DIM2>(
                (ValueType*) filteredData,
                Space(),
                header.simSize,
                header.simSize.x() * sizeof(ValueType)));
            MessageHeader mHeader;
            MessageHeader* fakeHeader = &mHeader;
            memcpy(fakeHeader, &header, sizeof(MessageHeader));

            char* recvHeader = new char[sizeof(MessageHeader) * numRanks];

            if(fullData == nullptr && mpiRank == 0)
                fullData = (char*) new ValueType[header.nodeSize.productOfComponents() * numRanks];

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Gather(
                fakeHeader,
                sizeof(MessageHeader),
                MPI_CHAR,
                recvHeader,
                sizeof(MessageHeader),
                MPI_CHAR,
                0,
                comm));

            const size_t elementsCount = header.nodeSize.productOfComponents() * sizeof(ValueType);

            MPI_CHECK(MPI_Gather(
                (char*) (data.getPointer()),
                elementsCount,
                MPI_CHAR,
                fullData,
                elementsCount,
                MPI_CHAR,
                0,
                comm));


            if(mpiRank == 0)
            {
                if(filteredData == nullptr)
                    filteredData = (char*) new ValueType[header.simSize.productOfComponents()];

                /*create box with valid memory*/
                dstBox = Box(PitchedBox<ValueType, DIM2>(
                    (ValueType*) filteredData,
                    Space(),
                    header.simSize,
                    header.simSize.x() * sizeof(ValueType)));


                for(int i = 0; i < numRanks; ++i)
                {
                    MessageHeader* head = (MessageHeader*) (recvHeader + sizeof(MessageHeader) * i);
                    size_t offset = header.nodeSize.productOfComponents() * static_cast<size_t>(i);
                    Box srcBox = Box(PitchedBox<ValueType, DIM2>(
                        reinterpret_cast<ValueType*>(fullData) + offset,
                        Space(),
                        head->nodeSize,
                        head->nodeSize.x() * sizeof(ValueType)));

                    insertData(dstBox, srcBox, head->nodeOffset, head->nodePictureSize, head->nodeGuardCells);
                }
            }

            delete[] recvHeader;

            return dstBox;
        }

        template<class DstBox, class SrcBox>
        void insertData(DstBox& dst, const SrcBox& src, Space offsetToSimNull, Space srcSize, Space nodeGuardCells)
        {
            for(int y = 0; y < srcSize.y(); ++y)
            {
                for(int x = 0; x < srcSize.x(); ++x)
                {
                    dst[y + offsetToSimNull.y()][x + offsetToSimNull.x()]
                        = src[nodeGuardCells.y() + y][nodeGuardCells.x() + x];
                }
            }
        }

    private:
        char* filteredData;
        char* fullData;
        MPI_Comm comm;
        int mpiRank;
        int numRanks;
        bool isMPICommInitialized;
        MessageHeader header;
    };

} // namespace gol
