/**
 * Copyright 2013-2015 Heiko Burau, Benjamin Worpitz, Alexander Grund
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

#include "mappings/simulation/GridController.hpp"
#include "cuSTL/container/copier/Memcopy.hpp"
#include "communication/manager_common.h"

#include <iostream>

namespace PMacc
{
namespace algorithm
{
namespace mpi
{

namespace GatherHelper
{

/** @tparam dim dimension of mpi cluster
 *  @tparam memDim dimension of memory to be gathered
 *
 * if memDim == dim - 1 then ``dir`` indicates the direction (orientation)
 * of the (meta)plane.
 */
template<int dim, int memDim>
struct posInMem;

template<int dim>
struct posInMem<dim, dim>
{
    math::Int<dim> operator()(const math::Int<dim>& pos, int) const
    {
        return pos;
    }
};

template<>
struct posInMem<DIM3, DIM2>
{
    math::Int<DIM2> operator()(const math::Int<DIM3>& pos, int dir) const
    {
        return math::Int<DIM2>(pos[(dir+1)%3], pos[(dir+2)%3]);
    }
};

template<>
struct posInMem<DIM2, DIM1>
{
    math::Int<DIM1> operator()(const math::Int<DIM2>& pos, int dir) const
    {
        return math::Int<DIM1>(pos[(dir+1)%2]);
    }
};

}

template<int dim>
Gather<dim>::Gather(const zone::SphericZone<dim>& p_zone) : comm(MPI_COMM_NULL)
{
    using namespace PMacc::math;

    PMacc::GridController<dim>& con = PMacc::Environment<dim>::get().GridController();
    Int<dim> pos = con.getPosition();

    int numWorldRanks; MPI_Comm_size(MPI_COMM_WORLD, &numWorldRanks);
    std::vector<Int<dim> > allPositions(numWorldRanks);

    MPI_CHECK(MPI_Allgather((void*)&pos, sizeof(Int<dim>), MPI_CHAR,
                  (void*)allPositions.data(), sizeof(Int<dim>), MPI_CHAR,
                  MPI_COMM_WORLD));

    std::vector<int> new_ranks;
    int myWorldId; MPI_Comm_rank(MPI_COMM_WORLD, &myWorldId);

    this->m_participate = false;
    for(int i = 0; i < (int)allPositions.size(); i++)
    {
        Int<dim> pos = allPositions[i];
        if(!p_zone.within(pos)) continue;

        new_ranks.push_back(i);
        this->positions.push_back(allPositions[i]);
        if(i == myWorldId) this->m_participate = true;
    }
    MPI_Group world_group, new_group;

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
        MPI_CHECK(MPI_Comm_free(&this->comm));
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
    int myId; MPI_Comm_rank(this->comm, &myId);
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
    int myId; MPI_Comm_rank(this->comm, &myId);
    return myId;
}

template<int dim>
template<typename Type, int memDim>
void Gather<dim>::CopyToDest<Type, memDim>::operator()(
                    const Gather<dim>& gather,
                    container::HostBuffer<Type, memDim>& dest,
                    std::vector<Type>& tmpDest,
                    container::HostBuffer<Type, memDim>& source, int dir) const
{
    using namespace math;

    for(int i = 0; i < (int)gather.positions.size(); i++)
    {
        Int<dim> pos = gather.positions[i];
        Int<memDim> posInMem = GatherHelper::posInMem<dim, memDim>()(pos, dir);

        cudaWrapper::Memcopy<memDim>()(&(*dest.origin()(posInMem * (Int<memDim>)source.size())), dest.getPitch(),
                                  tmpDest.data() + i * source.size().productOfComponents(), source.getPitch(),
                                  source.size(), cudaWrapper::flags::Memcopy::hostToHost);
    }
}

template<int dim>
template<typename Type, int memDim>
void Gather<dim>::operator()(container::HostBuffer<Type, memDim>& dest,
                             container::HostBuffer<Type, memDim>& source, int dir) const
{
    if(!this->m_participate) return;

    int numRanks; MPI_Comm_size(this->comm, &numRanks);
    std::vector<Type> tmpDest(root() ? numRanks * source.size().productOfComponents() : 0);

    MPI_CHECK(MPI_Gather((void*)source.getDataPointer(), source.size().productOfComponents() * sizeof(Type), MPI_CHAR,
               root() ? (void*)tmpDest.data() : NULL, source.size().productOfComponents() * sizeof(Type), MPI_CHAR,
               0, this->comm));
    if(!root()) return;

    CopyToDest<Type, memDim>()(*this, dest, tmpDest, source, dir);
}

} // mpi
} // algorithm
} // PMacc
