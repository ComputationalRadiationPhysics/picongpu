#pragma once

/**
 * Copyright 2013 Heiko Burau
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
class Reduce
{
private:
    MPI_Comm comm;
    bool m_participate;
public:
    Reduce(const zone::SphericZone<dim>& zone, bool setThisAsRoot = false);
    ~Reduce();
    
    template<typename Type, int conDim, typename ExprOrFunctor>
    void operator()(container::HostBuffer<Type, conDim>& dest, 
                    const container::HostBuffer<Type, conDim>& src,
                    ExprOrFunctor) const;
           
    inline bool participate() const {return m_participate;}
    inline bool root() const;
    inline int rank() const;
};
    
} // mpi
} // algorithm
} // PMacc

#include "cuSTL/algorithm/mpi/Reduce.tpp"
