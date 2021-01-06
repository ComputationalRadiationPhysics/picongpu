/* Copyright 2013-2021 Rene Widera
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
#include "pmacc/mpi/MPI_StructAsArray.hpp"

namespace pmacc
{
    namespace mpi
    {
        namespace def
        {
            template<typename Type>
            struct GetMPI_StructAsArray;

        } // namespace def

        template<typename Type>
        pmacc::mpi::MPI_StructAsArray getMPI_StructAsArray()
        {
            return def::GetMPI_StructAsArray<Type>()();
        }

    } // namespace mpi

} // namespace pmacc

#include "pmacc/mpi/GetMPI_StructAsArray.tpp"
