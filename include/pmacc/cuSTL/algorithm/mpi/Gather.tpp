/* Copyright 2013-2021 Heiko Burau, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/cuSTL/container/copier/Memcopy.hpp"

#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/communication/manager_common.hpp"

#include <iostream>
#include <numeric> // std::partial_sum
#include <algorithm> // std::copy


namespace pmacc
{
    namespace algorithm
    {
        namespace mpi
        {
            namespace GatherHelper
            {
                template<int dim, typename Type>
                struct ContiguousPitch
                {
                    math::Size_t<dim - 1> operator()(const math::Size_t<dim>& size)
                    {
                        math::Size_t<dim - 1> pitch;

                        pitch[0] = size[0] * sizeof(Type);
                        for(int axis = 1; axis < dim - 1; axis++)
                            pitch[axis] = pitch[axis - 1] * size[axis];

                        return pitch;
                    }
                };

                template<typename Type>
                struct ContiguousPitch<DIM1, Type>
                {
                    math::Size_t<0> operator()(const math::Size_t<DIM1>&)
                    {
                        return math::Size_t<0>();
                    }
                };

            } // namespace GatherHelper

            template<int dim>
            Gather<dim>::Gather(const zone::SphericZone<dim>& p_zone) : comm(MPI_COMM_NULL)
            {
                using namespace pmacc::math;

                pmacc::GridController<dim>& con = pmacc::Environment<dim>::get().GridController();
                Int<dim> pos = con.getPosition();

                int numWorldRanks;
                MPI_Comm_size(MPI_COMM_WORLD, &numWorldRanks);
                std::vector<Int<dim>> allPositions(numWorldRanks);

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    static_cast<void*>(&pos),
                    sizeof(Int<dim>),
                    MPI_CHAR,
                    static_cast<void*>(allPositions.data()),
                    sizeof(Int<dim>),
                    MPI_CHAR,
                    MPI_COMM_WORLD));

                std::vector<int> new_ranks;
                int myWorldId;
                MPI_Comm_rank(MPI_COMM_WORLD, &myWorldId);

                this->m_participate = false;
                for(int i = 0; i < static_cast<int>(allPositions.size()); i++)
                {
                    Int<dim> pos = allPositions[i];
                    if(!p_zone.within(pos))
                        continue;

                    new_ranks.push_back(i);
                    this->positions.push_back(allPositions[i]);
                    if(i == myWorldId)
                        this->m_participate = true;
                }
                MPI_Group world_group, new_group;

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
                MPI_CHECK(MPI_Group_incl(world_group, new_ranks.size(), new_ranks.data(), &new_group));
                MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, new_group, &this->comm));
                MPI_CHECK(MPI_Group_free(&new_group));
            }

            template<int dim>
            Gather<dim>::~Gather()
            {
                if(this->comm != MPI_COMM_NULL)
                {
                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    __getTransactionEvent().waitForFinished();
                    MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&this->comm));
                }
            }

            template<int dim>
            bool Gather<dim>::root() const
            {
                if(!this->m_participate)
                {
                    std::cerr << "error[mpi::Gather::root()]: this process does not participate in gathering.\n";
                    return false;
                }
                int myId;
                MPI_Comm_rank(this->comm, &myId);
                return myId == 0;
            }

            template<int dim>
            int Gather<dim>::rank() const
            {
                if(!this->m_participate)
                {
                    std::cerr << "error[mpi::Gather::rank()]: this process does not participate in gathering.\n";
                    return -1;
                }
                int myId;
                MPI_Comm_rank(this->comm, &myId);
                return myId;
            }

            template<int dim>
            template<typename Type, int memDim, class T_Alloc, class T_Copy, class T_Assign>
            void Gather<dim>::CopyToDest::operator()(
                const Gather<dim>& gather,
                container::CartBuffer<Type, memDim, T_Alloc, T_Copy, T_Assign>& dest,
                std::vector<Type>& tmpDest,
                int dir,
                const std::vector<math::Size_t<memDim>>& srcSizes,
                const std::vector<size_t>& srcOffsets1D) const
            {
                using namespace math;

                int numRanks = static_cast<int>(gather.positions.size());

                // calculate sizes per axis in destination buffer
                std::vector<size_t> sizesPerAxis[memDim];

                // sizes per axis
                for(int i = 0; i < numRanks; i++)
                {
                    Int<dim> pos = gather.positions[i];
                    Int<memDim> posInMem = pos.template shrink<memDim>(dir + 1);
                    for(int axis = 0; axis < memDim; axis++)
                    {
                        size_t posOnAxis = static_cast<size_t>(posInMem[axis]);
                        if(posOnAxis >= sizesPerAxis[axis].size())
                            sizesPerAxis[axis].resize(posOnAxis + 1);
                        sizesPerAxis[axis][posOnAxis] = srcSizes[i][axis];
                    }
                }

                // calculate offsets per axis in destination buffer
                std::vector<size_t> offsetsPerAxis[memDim];

                // offsets per axis
                for(int axis = 0; axis < memDim; axis++)
                {
                    offsetsPerAxis[axis].resize(sizesPerAxis[axis].size());
                    std::vector<size_t> partialSum(offsetsPerAxis[axis].size());
                    std::partial_sum(sizesPerAxis[axis].begin(), sizesPerAxis[axis].end(), partialSum.begin());
                    offsetsPerAxis[axis][0] = 0;
                    std::copy(partialSum.begin(), partialSum.end() - 1, offsetsPerAxis[axis].begin() + 1);
                }

                // copy from one dimensional mpi buffer to n dimensional destination buffer
                for(int i = 0; i < numRanks; i++)
                {
                    Int<dim> pos = gather.positions[i];
                    Int<memDim> posInMem = pos.template shrink<memDim>(dir + 1);
                    Int<memDim> ndim_offset;
                    for(int axis = 0; axis < memDim; axis++)
                        ndim_offset[axis] = offsetsPerAxis[axis][posInMem[axis]];

                    // calculate srcPitch (contiguous memory)
                    Size_t<memDim - 1> srcPitch = GatherHelper::ContiguousPitch<memDim, Type>()(srcSizes[i]);

                    cuplaWrapper::Memcopy<memDim>()(
                        &(*dest.origin()(ndim_offset)),
                        dest.getPitch(),
                        tmpDest.data() + srcOffsets1D[i],
                        srcPitch,
                        srcSizes[i],
                        cuplaWrapper::flags::Memcopy::hostToHost);
                }
            }

            template<int dim>
            template<
                typename Type,
                int memDim,
                class T_Alloc,
                class T_Copy,
                class T_Assign,
                class T_Alloc2,
                class T_Copy2,
                class T_Assign2>
            void Gather<dim>::operator()(
                container::CartBuffer<Type, memDim, T_Alloc, T_Copy, T_Assign>& dest,
                container::CartBuffer<Type, memDim, T_Alloc2, T_Copy2, T_Assign2>& source,
                int dir) const
            {
                using namespace pmacc::math;

                if(!this->m_participate)
                    return;
                typedef container::CartBuffer<Type, memDim, T_Alloc, T_Copy, T_Assign> DestBuffer;
                typedef container::CartBuffer<Type, memDim, T_Alloc2, T_Copy2, T_Assign2> SrcBuffer;
                PMACC_CASSERT_MSG(
                    Can_Only_Gather_Host_Memory,
                    boost::is_same<typename DestBuffer::memoryTag, allocator::tag::host>::value
                        && boost::is_same<typename SrcBuffer::memoryTag, allocator::tag::host>::value);

                const bool useTmpSrc = source.isContigousMemory();
                int numRanks;
                MPI_Comm_size(this->comm, &numRanks);
                std::vector<Type> tmpDest(root() ? numRanks * source.size().productOfComponents() : 0);
                container::HostBuffer<Type, memDim> tmpSrc(
                    useTmpSrc ? source.size() : math::Size_t<memDim>::create(0));
                if(useTmpSrc)
                    tmpSrc = source; /* Mem copy */

                // Get number of elements for each source buffer
                std::vector<Size_t<memDim>> srcBufferSizes(numRanks);
                Size_t<memDim> srcBufferSize = source.size();
                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Gather(
                    static_cast<void*>(&srcBufferSize),
                    sizeof(Size_t<memDim>),
                    MPI_CHAR,
                    static_cast<void*>(srcBufferSizes.data()),
                    sizeof(Size_t<memDim>),
                    MPI_CHAR,
                    0,
                    this->comm));

                // 1D offsets in destination buffer
                std::vector<size_t> srcBufferOffsets1D(numRanks);
                std::vector<size_t> srcBufferSizes1D(numRanks);
                std::vector<int> srcBufferOffsets1D_char(numRanks); // `MPI_Gatherv` demands `int*`
                std::vector<int> srcBufferSizes1D_char(numRanks);

                if(this->root())
                {
                    for(int i = 0; i < numRanks; i++)
                        srcBufferSizes1D[i] = srcBufferSizes[i].productOfComponents();
                    std::vector<size_t> partialSum(numRanks);
                    std::partial_sum(srcBufferSizes1D.begin(), srcBufferSizes1D.end(), partialSum.begin());
                    srcBufferOffsets1D[0] = 0;
                    std::copy(partialSum.begin(), partialSum.end() - 1, srcBufferOffsets1D.begin() + 1);

                    for(int i = 0; i < numRanks; i++)
                    {
                        srcBufferOffsets1D_char[i] = static_cast<int>(srcBufferOffsets1D[i]) * sizeof(Type);
                        srcBufferSizes1D_char[i] = static_cast<int>(srcBufferSizes1D[i]) * sizeof(Type);
                    }
                }

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                // gather
                MPI_CHECK(MPI_Gatherv(
                    useTmpSrc ? static_cast<void*>(tmpSrc.getDataPointer())
                              : static_cast<void*>(source.getDataPointer()),
                    source.size().productOfComponents() * sizeof(Type),
                    MPI_CHAR,
                    root() ? static_cast<void*>(tmpDest.data()) : nullptr,
                    srcBufferSizes1D_char.data(),
                    srcBufferOffsets1D_char.data(),
                    MPI_CHAR,
                    0,
                    this->comm));
                if(!root())
                    return;

                CopyToDest()(*this, dest, tmpDest, dir, srcBufferSizes, srcBufferOffsets1D);
            }

        } // namespace mpi
    } // namespace algorithm
} // namespace pmacc
