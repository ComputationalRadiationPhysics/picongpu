/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <mpi.h>

namespace pmacc
{
    namespace mpi
    {
        struct MPI_StructAsArray
        {
            MPI_StructAsArray(MPI_Datatype type, uint32_t factor) : dataType(type), sizeMultiplier(factor)
            {
            }

            MPI_Datatype dataType;
            uint32_t sizeMultiplier;
        };
    } // namespace mpi
} // namespace pmacc
