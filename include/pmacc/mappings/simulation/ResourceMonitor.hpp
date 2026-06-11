/*
 * SPDX-FileCopyrightText: Erik Zenker
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once
#include <cstdlib> /* std::size_t */
#include <vector> /* std::vector */

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
