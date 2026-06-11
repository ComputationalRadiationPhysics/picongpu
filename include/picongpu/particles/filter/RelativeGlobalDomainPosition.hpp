/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/filter/RelativeGlobalDomainPosition.def"

namespace picongpu
{
    namespace particles
    {
        namespace filter
        {
            namespace acc
            {
                template<typename T_Params>
                struct RelativeGlobalDomainPosition
                {
                    using Params = T_Params;

                    HDINLINE RelativeGlobalDomainPosition(
                        DataSpace<simDim> const& localDomainOffset,
                        DataSpace<simDim> const& globalDomainSize,
                        DataSpace<simDim> const& localSuperCellOffset)
                        : m_localDomainOffset(localDomainOffset)
                        , m_globalDomainSize(globalDomainSize)
                        , m_localSuperCellOffset(localSuperCellOffset)
                    {
                    }

                    template<typename T_Worker, typename T_Particle>
                    HDINLINE bool operator()(T_Worker const&, T_Particle const& particle)
                    {
                        if(particle.isHandleValid())
                        {
                            using SuperCellSize = typename T_Particle::SuperCellSize;
                            /* offset of the superCell (in cells, without any guards) to the origin of the global
                             * domain */
                            DataSpace<simDim> globalSuperCellOffset
                                = m_localDomainOffset + (m_localSuperCellOffset * SuperCellSize::toRT());
                            return isParticleInsideRange(particle, globalSuperCellOffset);
                        }
                        return false;
                    }

                private:
                    /** check if a particle is located in the user defined range
                     *
                     * @tparam T_Particle type of the particle
                     * @param particle particle than needs to be checked
                     * @param globalSuperCellOffset offset of the superCell (in cells, without any guards)
                     *                              to the origin of the global domain
                     */
                    template<typename T_Particle>
                    HDINLINE bool isParticleInsideRange(
                        T_Particle const& particle,
                        DataSpace<simDim> const& globalSuperCellOffset) const
                    {
                        using SuperCellSize = typename T_Particle::SuperCellSize;

                        int const particleCellIdx = particle[localCellIdx_];
                        DataSpace<simDim> const cellInSuperCell
                            = pmacc::math::mapToND(SuperCellSize::toRT(), particleCellIdx);
                        DataSpace<simDim> const globalParticleOffset(globalSuperCellOffset + cellInSuperCell);

                        float_X const relativePosition = float_X(globalParticleOffset[Params::dimension])
                                                         / float_X(m_globalDomainSize[Params::dimension]);

                        return (Params::lowerBound <= relativePosition && relativePosition < Params::upperBound);
                    }

                    DataSpace<simDim> const m_localDomainOffset;
                    DataSpace<simDim> const m_globalDomainSize;
                    DataSpace<simDim> const m_localSuperCellOffset;
                };

            } // namespace acc

            template<typename T_Params>
            struct RelativeGlobalDomainPosition
            {
                using Params = T_Params;

                template<typename T_SpeciesType>
                struct apply
                {
                    using type = RelativeGlobalDomainPosition;
                };

                HINLINE RelativeGlobalDomainPosition()
                {
                    SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                    globalDomainSize = subGrid.getGlobalDomain().size;
                    localDomainOffset = subGrid.getLocalDomain().offset;
                }

                /** create filter for the accelerator
                 *
                 * @tparam T_Worker lockstep::Worker, lockstep worker
                 * @param localSupercellOffset offset (in superCells, without any guards) relative
                 *                        to the origin of the local domain
                 * @param configuration of the worker
                 */
                template<typename T_Worker>
                HDINLINE acc::RelativeGlobalDomainPosition<Params> operator()(
                    T_Worker const& worker,
                    DataSpace<simDim> const& localSuperCellOffset) const
                {
                    return acc::RelativeGlobalDomainPosition<Params>(
                        localDomainOffset,
                        globalDomainSize,
                        localSuperCellOffset);
                }

                HINLINE static std::string getName()
                {
                    // we provide the name from the param class
                    return T_Params::name;
                }

                /** A filter is deterministic if the filter outcome is equal between evaluations. If so, set this
                 * variable to true, otherwise to false.
                 *
                 * Example: A filter were results depend on a random number generator must return false.
                 */
                static constexpr bool isDeterministic = true;

                DataSpace<simDim> localDomainOffset;
                DataSpace<simDim> globalDomainSize;
            };

        } // namespace filter
    } // namespace particles
} // namespace picongpu
