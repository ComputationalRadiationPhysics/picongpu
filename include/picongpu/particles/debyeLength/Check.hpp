/* Copyright 2020-2023 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::debyeLength
{
    /** Check Debye length resolution
     *
     * Compute and print the weighted average Debye length for the electron species.
     * Print in how many supercells the locally estimated Debye length is not resolved with a single cell.
     *
     * The check is supposed to be called just after the particles are initialized at start of a simulation.
     * The results are output to log<picLog::PHYSICS>.
     *
     * This function must be called from all MPI ranks.
     *
     * @param cellDescription mapping for kernels
     */
    void check(MappingDesc const cellDescription);

} // namespace picongpu::particles::debyeLength
