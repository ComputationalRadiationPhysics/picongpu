/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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


#ifndef GATHERSLICE_HPP
#define	GATHERSLICE_HPP

#include "types.h"


#include <mpi.h>
#include "mappings/simulation/GridController.hpp"

//c includes
#include "memory/boxes/PitchedBox.hpp"
#include "dimensions/DataSpace.hpp"

namespace gol
{
using namespace PMacc;

struct MessageHeader
{

    MessageHeader()
    {
    }

    MessageHeader(Space simSize, GridLayout<DIM2> layout, Space nodeOffset) :
    simSize(simSize),
    nodeOffset(nodeOffset)
    {
        nodeSize = layout.getDataSpace();
        nodePictureSize = layout.getDataSpaceWithoutGuarding();
        nodeGuardCells = layout.getGuard();
    }

    enum
    {
        bytes = 128
    };
    Space simSize;
    Space nodeSize;
    Space nodePictureSize;
    Space nodeGuardCells;
    Space nodeOffset;

};

struct GatherSlice
{

    GatherSlice() : mpiRank(-1), numRanks(0), filteredData(NULL), fullData(NULL), isMPICommInitialized(false)
    {
    }

    ~GatherSlice()
    {

    }

    void finalize()
    {
        if (filteredData != NULL)
        {  
            delete[] filteredData;
            filteredData=NULL;
        }
        if (fullData != NULL)
        {
            delete[] fullData;
            fullData=NULL;
        }
        if (isMPICommInitialized)
        {
            MPI_Comm_free(&comm);
            isMPICommInitialized=false;
        }
        mpiRank=-1;
    }

    /*
     * @return true if object has reduced data after reduce call else false
     */
    bool init(const MessageHeader mHeader, bool isActive)
    {
        header = mHeader;

        int countRanks = Environment<DIM2>::getInstance().GridController().getGpuNodes().productOfComponents();
        int gatherRanks[countRanks];
        int groupRanks[countRanks];
        mpiRank = Environment<DIM2>::getInstance().GridController().getGlobalRank();
        if (!isActive)
            mpiRank = -1;

        MPI_CHECK(MPI_Allgather(&mpiRank, 1, MPI_INT, gatherRanks, 1, MPI_INT, MPI_COMM_WORLD));

        for (int i = 0; i < countRanks; ++i)
        {
            if (gatherRanks[i] != -1)
            {
                groupRanks[numRanks] = gatherRanks[i];
                numRanks++;
            }
        }

        MPI_Group group;
        MPI_Group newgroup;
        MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
        MPI_CHECK(MPI_Group_incl(group, numRanks, groupRanks, &newgroup));

        MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &comm));

        if (mpiRank != -1)
        {
            MPI_Comm_rank(comm, &mpiRank);
            isMPICommInitialized = true;
        }

        return mpiRank == 0;
    }

    template<class Box >
    Box operator()(Box data)
    {
        typedef typename Box::ValueType ValueType;

        Box dstBox = Box(PitchedBox<ValueType, DIM2 > (
                                                       (ValueType*) filteredData,
                                                       Space(),
                                                       header.simSize,
                                                       header.simSize.x() * sizeof (ValueType)
                                                       ));
        MessageHeader mHeader;
        MessageHeader* fakeHeader = &mHeader;
        memcpy(fakeHeader, &header, MessageHeader::bytes);

        char* recvHeader = new char[ MessageHeader::bytes * numRanks];

        if (fullData == NULL && mpiRank == 0)
            fullData = (char*) new ValueType[header.nodeSize.productOfComponents() * numRanks];


        MPI_CHECK(MPI_Gather(fakeHeader, MessageHeader::bytes, MPI_CHAR, recvHeader, MessageHeader::bytes,
                             MPI_CHAR, 0, comm));

        const size_t elementsCount = header.nodeSize.productOfComponents() * sizeof (ValueType);

        MPI_CHECK(MPI_Gather(
                             (char*) (data.getPointer()), elementsCount, MPI_CHAR,
                             fullData, elementsCount, MPI_CHAR,
                             0, comm));



        if (mpiRank == 0)
        {
            if (filteredData == NULL)
                filteredData = (char*) new ValueType[header.simSize.productOfComponents()];

            /*create box with valid memory*/
            dstBox = Box(PitchedBox<ValueType, DIM2 > (
                                                       (ValueType*) filteredData,
                                                       Space(),
                                                       header.simSize,
                                                       header.simSize.x() * sizeof (ValueType)
                                                       ));


            for (int i = 0; i < numRanks; ++i)
            {
                MessageHeader* head = (MessageHeader*) (recvHeader + MessageHeader::bytes * i);
                Box srcBox = Box(PitchedBox<ValueType, DIM2 > (
                                                               (ValueType*) fullData,
                                                               Space(0, head->nodeSize.y() * i),
                                                               head->nodeSize,
                                                               head->nodeSize.x() * sizeof (ValueType)
                                                               ));

                insertData(dstBox, srcBox, head->nodeOffset, head->nodePictureSize, head->nodeGuardCells);
            }

        }

        delete[] recvHeader;

        return dstBox;
    }

    template<class DstBox, class SrcBox>
    void insertData(DstBox& dst, const SrcBox& src, Space offsetToSimNull, Space srcSize, Space nodeGuardCells)
    {
        for (int y = 0; y < srcSize.y(); ++y)
        {
            for (int x = 0; x < srcSize.x(); ++x)
            {
                dst[y + offsetToSimNull.y()][x + offsetToSimNull.x()] =
                    src[nodeGuardCells.y() + y][nodeGuardCells.x() + x];
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

}//namespace

#endif	/* GATHERSLICE_HPP */

