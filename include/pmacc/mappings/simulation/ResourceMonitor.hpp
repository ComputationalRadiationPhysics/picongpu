/* Copyright 2016-2021 Erik Zenker
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
#include <vector> /* std::vector */
#include <cstdlib> /* std::size_t */

namespace pmacc
{
    /**
     * Provides ressource information of the current subgrid
     *
     * @tparam T_DIM number of dimensions of the simulation
     */
    template<unsigned T_DIM>
    class ResourceMonitor
    {
    public:
        /**
         * Constructor
         */
        ResourceMonitor();

        /**
         *  Returns the number of cells on the device
         */
        std::size_t getCellCount();

        /**
         * Returns the number of particles per species on the device
         */
        template<typename T_Species, typename T_MappingDesc, typename T_ParticleFilter>
        std::vector<std::size_t> getParticleCounts(T_MappingDesc& cellDescription, T_ParticleFilter& parFilter);
    };

} // namespace pmacc
