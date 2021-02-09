/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/dimensions/GridLayout.hpp"
#include "Selection.hpp"

namespace pmacc
{
    /**
     * Groups local, global and total domain information.
     *
     * For a detailed description of domains, see the PIConGPU wiki page:
     * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
     */
    template<unsigned DIM>
    class SubGrid
    {
    public:
        typedef DataSpace<DIM> Size;

        constexpr SubGrid& operator=(const SubGrid&) = default;

        /**
         * Initialize SubGrid instance
         *
         * @param localSize local domain size
         * @param globalSize global domain size
         * @param localOffset local domain offset (formerly 'globalOffset')
         */
        void init(const Size& localSize, const Size& globalSize, const Size& localOffset)
        {
            totalDomain = Selection<DIM>(globalSize);
            globalDomain = Selection<DIM>(globalSize);
            localDomain = Selection<DIM>(localSize, localOffset);
        }

        /**
         * Set offset of the local domain.
         *
         * @param offset offset of local domain
         */
        void setLocalDomainOffset(const Size& offset)
        {
            localDomain = Selection<DIM>(localDomain.size, offset);
        }

        /**
         * Set offset of the global domain.
         *
         * @param offset offset of global domain
         */
        void setGlobalDomainOffset(const Size& offset)
        {
            globalDomain = Selection<DIM>(globalDomain.size, offset);
        }

        /**
         * Get the total domain
         *
         * total simulation volume, including active and inactive subvolumes
         *
         * @return selection for total domain
         */
        Selection<DIM> getTotalDomain() const
        {
            return totalDomain;
        }

        /**
         * Get the global domain
         *
         * currently simulated volume over all GPUs, offset relative to totalDomain
         *
         * @return selection for global domain
         */
        Selection<DIM> getGlobalDomain() const
        {
            return globalDomain;
        }

        /**
         * Get the local domain
         *
         * currently simulated volume on this GPU, offset relative to globalDomain
         *
         * @return selection for local domain
         */
        Selection<DIM> getLocalDomain() const
        {
            return localDomain;
        }

    private:
        friend class Environment<DIM>;

        /** total simulation volume, including active and inactive subvolumes */
        Selection<DIM> totalDomain;

        /** currently simulated volume over all GPUs, offset relative to totalDomain */
        Selection<DIM> globalDomain;

        /** currently simulated volume on this GPU, offset relative to globalDomain */
        Selection<DIM> localDomain;

        /**
         * Constructor
         */
        SubGrid()
        {
        }

        static SubGrid<DIM>& getInstance()
        {
            static SubGrid<DIM> instance;
            return instance;
        }

        virtual ~SubGrid()
        {
        }

        /**
         * Constructor
         */
        SubGrid(const SubGrid& gc)
        {
        }
    };


} // namespace pmacc
