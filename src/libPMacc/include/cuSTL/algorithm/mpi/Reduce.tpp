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
#include "lambda/make_Functor.hpp"

namespace PMacc
{
namespace algorithm
{
namespace mpi
{

template<int dim>
Reduce<dim>::Reduce(const zone::SphericZone<dim>& _zone)
{
    using namespace math;
    
    PMacc::GridController<dim>& con = PMacc::GridController<dim>::getInstance();
    PMacc::DataSpace<dim> _pos = con.getPosition();
    
    this->size = (UInt<dim>)_zone.size;
    this->pos.x() = _pos.x();
    this->pos.y() = _pos.y();
    this->pos.z() = _pos.z();
    int numWorldRanks; MPI_Comm_size(MPI_COMM_WORLD, &numWorldRanks);
    
    std::vector<Int<dim> > allPositions(numWorldRanks);
    MPI_Allgather((void*)&this->pos, sizeof(Int<dim>), MPI_CHAR,
                  (void*)allPositions.data(), sizeof(Int<dim>), MPI_CHAR,
                  MPI_COMM_WORLD);
                  
    std::vector<int> new_ranks;
    int myWorldId; MPI_Comm_rank(MPI_COMM_WORLD, &myWorldId);
    
    this->m_participate = false;
    for(int i = 0; i < (int)allPositions.size(); i++)
    {
        Int<dim> pos = allPositions[i];
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
    MPI_Group_free(&new_group);
}

template<int dim>
Reduce<dim>::~Reduce()
{
    if(this->comm != MPI_COMM_WORLD)
    {
        /*
        \todo: comm freigeben
        std::cout << "comm world: " << MPI_COMM_WORLD << ", comm: " << this->comm << std::endl;
        MPI_Comm_free(&this->comm);
        */
    }
}

template<int dim>
bool Reduce<dim>::root() const
{
    int myId; MPI_Comm_rank(this->comm, &myId);
    return myId == 0;
}

template<int dim>
int Reduce<dim>::rank() const
{
    int myId; MPI_Comm_rank(this->comm, &myId);
    return myId;
}

namespace detail
{
    
template<typename Functor, typename type>
struct MPI_User_Op
{
    static void callback(void* invec, void* inoutvec, int *len, MPI_Datatype*)
    {
        Functor functor;
        for(int i = 0; i < *len; i++)
        {
            ((type*)inoutvec)[i] = functor(((type*)inoutvec)[i], ((type*)invec)[i]);
        }
    }
};

} // detail

template<int dim>
template<typename Type, int conDim, typename ExprOrFunctor>
void Reduce<dim>::operator()
                   (container::HostBuffer<Type, conDim>& dest, 
                    const container::HostBuffer<Type, conDim>& src,
                    ExprOrFunctor) const
{
    if(!this->m_participate) return;
    
    typedef typename lambda::result_of::make_Functor<ExprOrFunctor>::type Functor;
    
    MPI_Op user_op;  
    MPI_Op_create(&detail::MPI_User_Op<Functor, Type>::callback, 1, &user_op);
    
    MPI_Reduce(&(*src.origin()), &(*dest.origin()), sizeof(Type) * dest.size().volume(),
        MPI_CHAR, user_op, 0, this->comm);
    
    MPI_Op_free(&user_op);
}

} // mpi
} // algorithm
} // PMacc
