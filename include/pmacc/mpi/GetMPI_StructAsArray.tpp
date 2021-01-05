/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace mpi
    {
        namespace def
        {
            template<>
            struct GetMPI_StructAsArray<int>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_INT, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<unsigned>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_UNSIGNED, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<long>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_LONG, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<unsigned long>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_UNSIGNED_LONG, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<long long>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_LONG_LONG, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<unsigned long long>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_UNSIGNED_LONG_LONG, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<float>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_FLOAT, 1);
                }
            };

            template<>
            struct GetMPI_StructAsArray<double>
            {
                MPI_StructAsArray operator()() const
                {
                    return MPI_StructAsArray(MPI_DOUBLE, 1);
                }
            };

        } // namespace def
    } // namespace mpi

} // namespace pmacc
