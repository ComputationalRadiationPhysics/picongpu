/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
#include "mappings/simulation/GridController.hpp"
#include <iostream>
#include "cuSTL/container/copier/Memcopy.hpp"

namespace PMacc
{
namespace algorithm
{
namespace mpi
{

template<int dim>
Gather<dim>::Gather(const zone::SphericZone<dim>& _zone) : comm(0)
{
    using namespace PMacc::math;
    
    PMacc::GridController<dim>& con = PMacc::GridController<dim>::getInstance();
    PMacc::DataSpace<dim> _pos = con.getPosition();
    
    this->size = (PMacc::math::UInt<dim>)_zone.size;
    this->pos.x() = _pos.x();
    this->pos.y() = _pos.y();
    this->pos.z() = _pos.z();
    int numWorldRanks; MPI_Comm_size(MPI_COMM_WORLD, &numWorldRanks);
    std::vector<PMacc::math::Int<dim> > allPositions(numWorldRanks);
    MPI_Allgather((void*)&this->pos, sizeof(Int<dim>), MPI_CHAR, 
                  (void*)allPositions.data(), sizeof(PMacc::math::Int<dim>), MPI_CHAR,
                  MPI_COMM_WORLD);
                  
    std::vector<int> new_ranks;
    int myWorldId; MPI_Comm_rank(MPI_COMM_WORLD, &myWorldId);
    this->m_participate = false;
    for(int i = 0; i < (int)allPositions.size(); i++)
    {
        PMacc::math::Int<dim> pos = allPositions[i];
        if(pos.x() < (int)_zone.offset.x() || pos.x() >= (int)_zone.offset.x() + (int)_zone.size.x()) continue;
        if(pos.y() < (int)_zone.offset.y() || pos.y() >= (int)_zone.offset.y() + (int)_zone.size.y()) continue;
        if(pos.z() < (int)_zone.offset.z() || pos.z() >= (int)_zone.offset.z() + (int)_zone.size.z()) continue;
        new_ranks.push_back(i);
        this->positions.push_back(allPositions[i]);
        if(i == myWorldId) this->m_participate = true;
    }
    
    MPI_Group world_group, new_group;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, new_ranks.size(), new_ranks.data(), &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &this->comm);
    //std::cout << "comm created: " << this->comm << std::endl;
    MPI_Group_free(&new_group);
}

template<int dim>
Gather<dim>::~Gather()
{
    if(this->comm != MPI_COMM_WORLD && this->comm != 0)
    {
        //\todo: comm freigeben
        //std::cout << "comm world: " << MPI_COMM_WORLD << ", comm: " << this->comm << std::endl;
        //MPI_Comm_free(&this->comm);
    }
}

template<int dim>
bool Gather<dim>::root() const
{
    int myId; MPI_Comm_rank(this->comm, &myId);
    return myId == 0;
}

template<int dim>
int Gather<dim>::rank() const
{
    int myId; MPI_Comm_rank(this->comm, &myId);
    return myId;
}

template<int dim>
template<typename Type>
void Gather<dim>::CopyToDest<Type, 3, 2>::operator()(
                    const Gather<dim>& gather,
                    container::HostBuffer<Type, 2>& dest,
                    std::vector<Type>& tmpDest,
                    container::HostBuffer<Type, 2>& source, int dir) const
{
    using namespace math;
    
    for(int i = 0; i < (int)gather.positions.size(); i++)
    {
        Int<3> pos = gather.positions[i];
        Int<2> pos2D(pos[(dir+1)%3], pos[(dir+2)%3]);
        
        cudaWrapper::Memcopy<2>()(&(*dest.origin()(pos2D * (Int<2>)source.size())), dest.getPitch(),
                                  tmpDest.data() + i * source.size().volume(), source.getPitch(),
                                  source.size(), cudaWrapper::flags::Memcopy::hostToHost);
    }
}

template<>
template<typename Type, int memDim>
void Gather<3>::operator()(container::HostBuffer<Type, memDim>& dest,
                             container::HostBuffer<Type, memDim>& source, int dir) const
{
    if(!this->m_participate) return;
    
    int numRanks; MPI_Comm_size(this->comm, &numRanks);
    std::vector<Type> tmpDest(numRanks * source.size().volume());
    MPI_Gather((void*)source.getDataPointer(), source.size().volume() * sizeof(Type), MPI_CHAR,
               (void*)tmpDest.data(), source.size().volume() * sizeof(Type), MPI_CHAR,
               0, this->comm);
    if(!root()) return;

    CopyToDest<Type, 3, memDim>()(*this, dest, tmpDest, source, dir);
}

} // mpi
} // algorithm
} // PMacc
