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
 
#ifndef ALGORITHM_MPI_GATHER_HPP
#define ALGORITHM_MPI_GATHER_HPP

#include "mpi.h"
#include "math/vector/Int.hpp"
#include "math/vector/UInt.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include <vector>

namespace PMacc
{
namespace algorithm
{
namespace mpi
{
    
template<int dim>
class Gather
{
private:
    MPI_Comm comm;
    std::vector<math::Int<dim> > positions;
    math::Int<dim> pos;
    math::UInt<dim> size;
    bool m_participate;
    
    template<typename Type, int _dim, int memDim>
    struct CopyToDest;
    
    template<typename Type>
    struct CopyToDest<Type, 3, 2>
    {
        void operator()(const Gather<dim>& gather,
                        container::HostBuffer<Type, 2>& dest,
                        std::vector<Type>& tmpDest,
                        container::HostBuffer<Type, 2>& source, int dir) const;
    };
    
    template<typename Type, int _dim, int memDim>
    friend class CopyToDest;
public:    
    Gather(const zone::SphericZone<dim>& _zone);
    ~Gather();
    
    template<typename Type, int memDim>
    void operator()(container::HostBuffer<Type, memDim>& dest,
                    container::HostBuffer<Type, memDim>& source,
                    int dir = -1) const;
                    
    inline bool participate() const {return m_participate;}
    inline bool root() const;
    inline int rank() const;
};

} // mpi
} // algorithm
} // PMacc

#include "Gather.tpp"

#endif // ALGORITHM_MPI_GATHER_HPP
