/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/mpi/MPI_StructAsArray.hpp"
#include "pmacc/types.hpp"

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
        HINLINE pmacc::mpi::MPI_StructAsArray getMPI_StructAsArray()
        {
            return def::GetMPI_StructAsArray<Type>()();
        }

    } // namespace mpi

} // namespace pmacc

#include "pmacc/mpi/GetMPI_StructAsArray.tpp"
