/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                    return {MPI_INT, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<unsigned>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_UNSIGNED, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<long>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_LONG, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<unsigned long>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_UNSIGNED_LONG, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<long long>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_LONG_LONG, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<unsigned long long>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_UNSIGNED_LONG_LONG, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<float>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_FLOAT, 1};
                }
            };

            template<>
            struct GetMPI_StructAsArray<double>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_DOUBLE, 1};
                }
            };

        } // namespace def
    } // namespace mpi

} // namespace pmacc
