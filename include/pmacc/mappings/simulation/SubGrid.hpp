/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "Selection.hpp"
#include "pmacc/Environment.def"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/GridLayout.hpp"

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
        using Size = DataSpace<DIM>;

        constexpr SubGrid& operator=(SubGrid const&) = default;

        /**
         * Initialize SubGrid instance
         *
         * @param localSize local domain size
         * @param globalSize global domain size
         * @param localOffset local domain offset (formerly 'globalOffset')
         */
        void init(Size const& localSize, Size const& globalSize, Size const& localOffset)
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
        void setLocalDomainOffset(Size const& offset)
        {
            localDomain = Selection<DIM>(localDomain.size, offset);
        }

        /**
         * Set offset of the global domain.
         *
         * @param offset offset of global domain
         */
        void setGlobalDomainOffset(Size const& offset)
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
        SubGrid() = default;

        static SubGrid<DIM>& getInstance()
        {
            static SubGrid<DIM> instance;
            return instance;
        }

        virtual ~SubGrid() = default;

        /**
         * Constructor
         */
        SubGrid(SubGrid const& gc)
        {
        }
    };


} // namespace pmacc
