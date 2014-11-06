/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#include "types.h"

namespace PMacc
{
namespace mpi
{
namespace def
{

template<>
struct GetMPI_StructAsArray<float3 >
{

    MPI_StructAsArray operator()() const
    {
        return MPI_StructAsArray(MPI_FLOAT, 3);
    }
};

template<>
struct GetMPI_StructAsArray<int >
{

    MPI_StructAsArray operator()() const
    {
        return MPI_StructAsArray(MPI_INT, 1);
    }
};

template<>
struct GetMPI_StructAsArray<float >
{

    MPI_StructAsArray operator()() const
    {
        return MPI_StructAsArray(MPI_FLOAT, 1);
    }
};

template<>
struct GetMPI_StructAsArray<uint64_cu >
{

    MPI_StructAsArray operator()() const
    {
        return MPI_StructAsArray(MPI_UNSIGNED_LONG_LONG, 1);
    }
};

template<>
struct GetMPI_StructAsArray<double >
{

    MPI_StructAsArray operator()() const
    {
        return MPI_StructAsArray(MPI_DOUBLE, 1);
    }
};

} //namespace def
}//namespace mpi

}//namespace PMacc

