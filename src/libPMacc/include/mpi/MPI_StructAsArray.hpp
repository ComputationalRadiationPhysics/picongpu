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

#ifndef MPI_STRUCTASARRAY_HPP
#define	MPI_STRUCTASARRAY_HPP

#include "types.h"
#include <mpi.h>

namespace PMacc
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
    }
}

#endif	/* MPI_STRUCTASARRAY_HPP */

