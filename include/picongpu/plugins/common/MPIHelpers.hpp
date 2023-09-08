/* Copyright 2021-2023 Franz Poeschel
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <string>
#include <vector>

#include <mpi.h>

namespace picongpu
{
    /**
     * @brief Helper class to help figure out a platform-independent
     *        MPI_Datatype for size_t.
     */
    template<typename>
    struct MPI_Types;

    template<>
    struct MPI_Types<unsigned long>
    {
        // can't make this constexpr due to MPI
        // so, make this non-static for simplicity
        MPI_Datatype value = MPI_UNSIGNED_LONG;
    };

    template<>
    struct MPI_Types<unsigned long long>
    {
        MPI_Datatype value = MPI_UNSIGNED_LONG_LONG;
    };

    template<>
    struct MPI_Types<unsigned>
    {
        MPI_Datatype value = MPI_UNSIGNED;
    };

    /**
     * @brief Read a file in MPI-collective manner.
     *
     * The file is read on rank 0 and its contents subsequently distributed
     * to all other ranks.
     *
     * @param path Path for the file to read.
     * @param comm MPI communicator.
     * @return std::string Full file content.
     */
    std::string collective_file_read(std::string const& path, MPI_Comm comm);
} // namespace picongpu
